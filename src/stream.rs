use crate::error::{ChatKitError, StreamingError, ToolCallError};
use crate::messages::{AssistantMessage, TextContent, TokenUsage, ToolContent, extract_thinking, reject_empty};
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::VecDeque;
use std::fmt;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::Mutex;
use tokio::time::Instant;

type BoxedStream = Pin<Box<dyn Stream<Item = Result<AssistantMessage, StreamingError>> + Send>>;

/// Clonable handle to a streaming LLM response; the inner stream is shared via `Arc<Mutex>`.
pub struct StreamResponse(Arc<Mutex<BoxedStream>>);

impl StreamResponse {
    #[must_use]
    pub fn new(stream: BoxedStream) -> Self {
        StreamResponse(Arc::new(Mutex::new(stream)))
    }

    pub async fn next(&self) -> Option<Result<AssistantMessage, StreamingError>> {
        let mut stream = self.0.lock().await;
        stream.next().await
    }
}

impl fmt::Debug for StreamResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MessageStream(...)")
    }
}

impl Clone for StreamResponse {
    fn clone(&self) -> Self {
        StreamResponse(Arc::clone(&self.0))
    }
}

#[cfg_attr(not(feature = "metrics"), allow(unused_variables))]
pub(crate) fn process_stream(
    stream: impl Stream<
        Item = Result<async_openai::types::chat::CreateChatCompletionStreamResponse, async_openai::error::OpenAIError>,
    > + Unpin
    + Send
    + 'static,
    start_time: Instant,
    service: String,
    model: Cow<'static, str>,
) -> BoxedStream {
    ProcessedStream::new(stream, start_time, service, model).boxed()
}

#[cfg_attr(not(feature = "metrics"), allow(dead_code))]
struct ProcessedStream<S> {
    stream: S,
    start_time: Instant,
    service: String,
    model: Cow<'static, str>,
    state: ProcessedStreamState,
    buffer: String,
    previous_tool_call_id: Option<String>,
    pending: VecDeque<AssistantMessage>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ProcessedStreamState {
    WaitingForFirstToken,
    StreamingText,
    StreamingThinking,
    Finished,
}

impl<S> ProcessedStream<S> {
    fn new(stream: S, start_time: Instant, service: String, model: Cow<'static, str>) -> Self {
        Self {
            stream,
            start_time,
            service,
            model,
            state: ProcessedStreamState::WaitingForFirstToken,
            buffer: String::new(),
            previous_tool_call_id: None,
            pending: VecDeque::new(),
        }
    }

    fn emit_message(&mut self, first_message: &mut Option<AssistantMessage>, message: AssistantMessage) {
        if first_message.is_none() {
            *first_message = Some(message);
        } else {
            self.pending.push_back(message);
        }
    }

    #[cfg(feature = "metrics")]
    fn record_first_token(&self) {
        // The precision loss is fine here, as we are only using it for metrics.
        // TODO use as_millis_f64() once it is stable
        #[allow(clippy::cast_precision_loss)]
        metrics::histogram!(
            "llm_time_to_first_token_ms",
            "service" => self.service.clone(),
            "model" => self.model.clone(),
        )
        .record(self.start_time.elapsed().as_millis() as f64);
    }

    #[cfg(feature = "metrics")]
    fn record_last_token(&self) {
        // The precision loss is fine here, as we are only using it for metrics.
        // TODO use as_millis_f64() once it is stable
        #[allow(clippy::cast_precision_loss)]
        metrics::histogram!(
            "llm_time_to_last_token_ms",
            "service" => self.service.clone(),
            "model" => self.model.clone(),
        )
        .record(self.start_time.elapsed().as_millis() as f64);
    }
}

impl<S> ProcessedStream<S>
where
    S: Stream<
            Item = Result<
                async_openai::types::chat::CreateChatCompletionStreamResponse,
                async_openai::error::OpenAIError,
            >,
        > + Unpin,
{
    fn process_chat_chunk(
        &mut self,
        chunk: async_openai::types::chat::CreateChatCompletionStreamResponse,
    ) -> Result<Option<AssistantMessage>, StreamingError> {
        let mut first_message = None;
        let tokens = chunk.usage.as_ref().map(|usage| TokenUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        });

        let Some(first) = chunk.choices.into_iter().next() else {
            return Ok(None);
        };

        if self.state == ProcessedStreamState::WaitingForFirstToken
            && (first.delta.content.is_some() || first.delta.tool_calls.is_some())
        {
            self.state = ProcessedStreamState::StreamingText;
            #[cfg(feature = "metrics")]
            self.record_first_token();
        }

        if let Some(tool_calls) = first.delta.tool_calls {
            let mut processed_tool_calls = Vec::new();
            for tc in tool_calls {
                let tool_call = process_tool_call_chunk(self.previous_tool_call_id.clone(), tc)?;
                self.previous_tool_call_id = Some(tool_call.id.clone());
                processed_tool_calls.push(tool_call);
            }

            self.emit_message(
                &mut first_message,
                AssistantMessage {
                    name: None,
                    text: None,
                    tools: processed_tool_calls,
                    tokens,
                },
            );
        } else if let Some(refusal) = first.delta.refusal {
            self.emit_message(
                &mut first_message,
                AssistantMessage {
                    name: None,
                    text: Some(TextContent {
                        text: None,
                        thinking: None,
                        refusal: Some(refusal),
                    }),
                    tools: Vec::new(),
                    tokens,
                },
            );
        } else if let Some(content) = first.delta.content {
            first_message = self.process_content(content);
        }

        Ok(first_message)
    }

    fn process_content(&mut self, content: String) -> Option<AssistantMessage> {
        let mut first_message = None;
        self.buffer.push_str(&content);

        loop {
            if self.state != ProcessedStreamState::StreamingThinking {
                if let Some(pos) = self.buffer.find("<think>") {
                    let text = self.buffer[..pos].to_string();
                    self.buffer.drain(..pos + 7);
                    self.state = ProcessedStreamState::StreamingThinking;

                    if let Some(text) = reject_empty(text) {
                        self.emit_message(
                            &mut first_message,
                            AssistantMessage {
                                name: None,
                                text: Some(TextContent {
                                    text: Some(text),
                                    thinking: None,
                                    refusal: None,
                                }),
                                tools: Vec::new(),
                                tokens: None,
                            },
                        );
                    }
                } else {
                    // No <think> tag found. Yield everything up to a possible partial tag.
                    if let Some(last_lt) = self.buffer.rfind('<') {
                        let remaining = &self.buffer[last_lt..];
                        if "<think>".starts_with(remaining) {
                            let to_yield = self.buffer[..last_lt].to_string();
                            self.buffer.drain(..last_lt);
                            if let Some(text) = reject_empty(to_yield) {
                                self.emit_message(
                                    &mut first_message,
                                    AssistantMessage {
                                        name: None,
                                        text: Some(TextContent {
                                            text: Some(text),
                                            thinking: None,
                                            refusal: None,
                                        }),
                                        tools: Vec::new(),
                                        tokens: None,
                                    },
                                );
                            }
                            break;
                        }
                    }

                    let to_yield = self.buffer.clone();
                    self.buffer.clear();
                    if let Some(text) = reject_empty(to_yield) {
                        self.emit_message(
                            &mut first_message,
                            AssistantMessage {
                                name: None,
                                text: Some(TextContent {
                                    text: Some(text),
                                    thinking: None,
                                    refusal: None,
                                }),
                                tools: Vec::new(),
                                tokens: None,
                            },
                        );
                    }
                    break;
                }
            } else if let Some(pos) = self.buffer.find("</think>") {
                let thinking = self.buffer[..pos].to_string();
                self.buffer.drain(..pos + 8);
                self.state = ProcessedStreamState::StreamingText;

                self.emit_message(
                    &mut first_message,
                    AssistantMessage {
                        name: None,
                        text: Some(TextContent {
                            text: None,
                            thinking: reject_empty(thinking),
                            refusal: None,
                        }),
                        tools: Vec::new(),
                        tokens: None,
                    },
                );
            } else {
                // No </think> tag found. Yield everything up to a possible partial tag.
                if let Some(last_lt) = self.buffer.rfind('<') {
                    let remaining = &self.buffer[last_lt..];
                    if "</think>".starts_with(remaining) {
                        let to_yield = self.buffer[..last_lt].to_string();
                        self.buffer.drain(..last_lt);
                        if let Some(thinking) = reject_empty(to_yield) {
                            self.emit_message(
                                &mut first_message,
                                AssistantMessage {
                                    name: None,
                                    text: Some(TextContent {
                                        text: None,
                                        thinking: Some(thinking),
                                        refusal: None,
                                    }),
                                    tools: Vec::new(),
                                    tokens: None,
                                },
                            );
                        }
                        break;
                    }
                }

                let to_yield = self.buffer.clone();
                self.buffer.clear();
                if let Some(thinking) = reject_empty(to_yield) {
                    self.emit_message(
                        &mut first_message,
                        AssistantMessage {
                            name: None,
                            text: Some(TextContent {
                                text: None,
                                thinking: Some(thinking),
                                refusal: None,
                            }),
                            tools: Vec::new(),
                            tokens: None,
                        },
                    );
                }
                break;
            }
        }

        first_message
    }
}

impl<S> Stream for ProcessedStream<S>
where
    S: Stream<
            Item = Result<
                async_openai::types::chat::CreateChatCompletionStreamResponse,
                async_openai::error::OpenAIError,
            >,
        > + Unpin,
{
    type Item = Result<AssistantMessage, StreamingError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(message) = self.pending.pop_front() {
            return Poll::Ready(Some(Ok(message)));
        }

        if self.state == ProcessedStreamState::Finished {
            return Poll::Ready(None);
        }

        loop {
            match Pin::new(&mut self.stream).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => match self.process_chat_chunk(chunk) {
                    Ok(Some(message)) => return Poll::Ready(Some(Ok(message))),
                    Ok(None) => {}
                    Err(error) => {
                        self.state = ProcessedStreamState::Finished;
                        return Poll::Ready(Some(Err(error)));
                    }
                },
                Poll::Ready(Some(Err(error))) => {
                    self.state = ProcessedStreamState::Finished;
                    return Poll::Ready(Some(Err(Box::new(error))));
                }
                Poll::Ready(None) => {
                    self.state = ProcessedStreamState::Finished;
                    #[cfg(feature = "metrics")]
                    self.record_last_token();
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

pub(crate) fn process_tool_call_chunk(
    previous_id: Option<String>,
    value: async_openai::types::chat::ChatCompletionMessageToolCallChunk,
) -> Result<ToolContent, ChatKitError> {
    let async_openai::types::chat::ChatCompletionMessageToolCallChunk { id, function, .. } = value;

    if let Some(async_openai::types::chat::FunctionCallStream {
        name: Some(name),
        arguments: Some(arguments),
    }) = function
    {
        let (thinking, arguments) = extract_thinking(&arguments);
        tracing::debug!(arguments = &arguments, "cleaned function call arguments");

        let arguments = Value::from_str(&arguments)?;

        let id = id
            .or(previous_id)
            .ok_or(ChatKitError::ToolCall(ToolCallError::MissingToolId))?;

        Ok(ToolContent {
            id,
            name,
            thinking,
            arguments,
        })
    } else {
        Err(ChatKitError::EmptyResponse)
    }
}

#[cfg(test)]
mod tests {
    use super::{process_stream, process_tool_call_chunk};
    use async_openai::types::chat::{ChatCompletionMessageToolCallChunk, CreateChatCompletionStreamResponse};
    use futures::{StreamExt, stream};
    use tokio::time::Instant;

    #[test]
    fn test_streaming_with_tools() {
        let json = r#"{
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "test_tool",
                "arguments": "{\"arg\": \"<think>streaming tool thoughts</think>done\"}"
            }
        }"#;
        let chunk: ChatCompletionMessageToolCallChunk = serde_json::from_str(json).unwrap();

        let response = process_tool_call_chunk(None, chunk).unwrap();
        assert_eq!(response.name, "test_tool");
        assert_eq!(response.thinking, Some("streaming tool thoughts".to_string()));
        assert_eq!(response.arguments["arg"], "done");
    }

    #[tokio::test]
    async fn test_streaming_no_tools() {
        let chunks = vec![
            Ok(serde_json::from_str::<CreateChatCompletionStreamResponse>(
                r#"{
                "id": "1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "Hello <thi"
                        }
                    }
                ]
            }"#,
            )
            .unwrap()),
            Ok(serde_json::from_str::<CreateChatCompletionStreamResponse>(
                r#"{
                "id": "2",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "nk>thought</think>world"
                        }
                    }
                ]
            }"#,
            )
            .unwrap()),
        ];

        let stream = stream::iter(chunks);
        let mut processed = process_stream(stream, Instant::now(), "test".to_string(), "test".into());

        let msg1 = processed.next().await.unwrap().unwrap();
        let text1 = msg1.text.unwrap();
        assert_eq!(text1.text, Some("Hello ".to_string()));
        assert_eq!(text1.thinking, None);

        let msg2 = processed.next().await.unwrap().unwrap();
        let text2 = msg2.text.unwrap();
        assert_eq!(text2.text, None);
        assert_eq!(text2.thinking, Some("thought".to_string()));

        let msg3 = processed.next().await.unwrap().unwrap();
        let text3 = msg3.text.unwrap();
        assert_eq!(text3.text, Some("world".to_string()));
        assert_eq!(text3.thinking, None);
    }
}

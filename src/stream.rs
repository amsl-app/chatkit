use crate::error::{ChatKitError, StreamingError, ToolCallError};
use crate::messages::{AssistantMessage, TokenUsage, ToolContent};
use crate::metrics::{MetricsRecorder, MetricsRecorderTrait};
use crate::utils::{extract_thinking, reject_empty};
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::collections::VecDeque;
use std::fmt;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::Mutex;

pub(crate) type BoxedStream = Pin<Box<dyn Stream<Item = Result<AssistantMessage, StreamingError>> + Send>>;

const THINK_OPEN_TAG: &str = "<think>";
const THINK_CLOSE_TAG: &str = "</think>";

type ToolCallChunk = async_openai::types::chat::ChatCompletionMessageToolCallChunk;

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

pub(crate) fn process_stream(
    stream: impl Stream<
        Item = Result<async_openai::types::chat::CreateChatCompletionStreamResponse, async_openai::error::OpenAIError>,
    > + Unpin
    + Send
    + 'static,
    metrics: MetricsRecorder,
) -> BoxedStream {
    ProcessedStream::new(stream, metrics).boxed()
}

struct ProcessedStream<S> {
    stream: S,
    metrics: MetricsRecorder,
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
    fn new(stream: S, metrics: MetricsRecorder) -> Self {
        Self {
            stream,
            metrics,
            state: ProcessedStreamState::WaitingForFirstToken,
            buffer: String::new(),
            previous_tool_call_id: None,
            pending: VecDeque::with_capacity(1),
        }
    }

    fn emit_message(&mut self, message: AssistantMessage) {
        self.pending.push_back(message);
    }

    fn emit_text(&mut self, text: String) {
        if let Some(text) = reject_empty(text) {
            self.emit_message(AssistantMessage::text(text));
        }
    }

    fn emit_thinking(&mut self, thinking: String) {
        if let Some(thinking) = reject_empty(thinking) {
            self.emit_message(AssistantMessage::thinking(Some(thinking)));
        }
    }

    fn emit_thinking_boundary(&mut self, thinking: String) {
        self.emit_message(AssistantMessage::thinking(reject_empty(thinking)));
    }

    fn process_until_tag(
        &mut self,
        tag: &str,
        next_state: ProcessedStreamState,
        emit_found: fn(&mut Self, String),
        emit_pending: fn(&mut Self, String),
    ) -> ControlFlow<()> {
        if let Some(pos) = self.buffer.find(tag) {
            let content = self.buffer[..pos].to_string();
            self.buffer.drain(..pos + tag.len());
            self.state = next_state;
            emit_found(self, content);
            return ControlFlow::Continue(());
        }

        if let Some(last_lt) = self.buffer.rfind('<') {
            let remaining = &self.buffer[last_lt..];
            if tag.starts_with(remaining) {
                let content = self.buffer[..last_lt].to_string();
                self.buffer.drain(..last_lt);
                emit_pending(self, content);
                return ControlFlow::Break(());
            }
        }

        let content = std::mem::take(&mut self.buffer);
        emit_pending(self, content);
        ControlFlow::Break(())
    }

    fn finish(&mut self) {
        self.state = ProcessedStreamState::Finished;
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
    ) -> Result<(), StreamingError> {
        let tokens = chunk.usage.as_ref().map(|usage| TokenUsage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        });

        let Some(first) = chunk.choices.into_iter().next() else {
            return Ok(());
        };

        if self.state == ProcessedStreamState::WaitingForFirstToken
            && (first.delta.content.is_some() || first.delta.tool_calls.is_some())
        {
            self.metrics.first_token();
        }

        if let Some(tool_calls) = first.delta.tool_calls {
            self.state = ProcessedStreamState::StreamingText;
            let mut processed_tool_calls = Vec::new();
            for tc in tool_calls {
                processed_tool_calls.push(self.process_tool_call_chunk(tc)?);
            }

            self.emit_message(AssistantMessage::tools(processed_tool_calls, tokens));
        } else if let Some(refusal) = first.delta.refusal {
            self.emit_message(AssistantMessage::refusal(refusal, tokens));
        } else if let Some(content) = first.delta.content {
            self.state = ProcessedStreamState::StreamingText;
            self.process_content(&content);
        }

        Ok(())
    }

    fn process_content(&mut self, content: &str) {
        self.buffer.push_str(content);

        loop {
            let control_flow = if self.state == ProcessedStreamState::StreamingThinking {
                self.process_until_tag(
                    THINK_CLOSE_TAG,
                    ProcessedStreamState::StreamingText,
                    Self::emit_thinking_boundary,
                    Self::emit_thinking,
                )
            } else {
                self.process_until_tag(
                    THINK_OPEN_TAG,
                    ProcessedStreamState::StreamingThinking,
                    Self::emit_text,
                    Self::emit_text,
                )
            };

            if control_flow.is_break() {
                break;
            }
        }
    }

    fn process_tool_call_chunk(&mut self, value: ToolCallChunk) -> Result<ToolContent, ChatKitError> {
        let (id, name, arguments) = tool_call_parts(value)?;

        let (thinking, arguments) = extract_thinking(arguments);
        tracing::debug!(arguments = &arguments, "cleaned function call arguments");

        let arguments = serde_json::from_str::<Value>(&arguments)?;

        let id = id
            .or_else(|| self.previous_tool_call_id.take())
            .ok_or(ChatKitError::ToolCall(ToolCallError::MissingToolId))?;
        self.previous_tool_call_id = Some(id.clone());

        Ok(ToolContent {
            id,
            name,
            thinking,
            arguments,
        })
    }
}

fn tool_call_parts(value: ToolCallChunk) -> Result<(Option<String>, String, String), ChatKitError> {
    let ToolCallChunk { id, function, .. } = value;
    if let Some(function) = function
        && let async_openai::types::chat::FunctionCallStream { name, arguments } = function
        && let Some(name) = name
        && let Some(arguments) = arguments
    {
        return Ok((id, name, arguments));
    }
    Err(ChatKitError::EmptyResponse)
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
                    Ok(()) => {
                        if let Some(message) = self.pending.pop_front() {
                            return Poll::Ready(Some(Ok(message)));
                        }
                    }
                    Err(error) => {
                        self.finish();
                        return Poll::Ready(Some(Err(error)));
                    }
                },
                Poll::Ready(Some(Err(error))) => {
                    self.finish();
                    return Poll::Ready(Some(Err(Box::new(error))));
                }
                Poll::Ready(None) => {
                    self.finish();
                    self.metrics.last_token();
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ProcessedStream, process_stream};
    use crate::config::CallConfig;
    use crate::metrics::MetricsRecorder;
    use async_openai::types::chat::{ChatCompletionMessageToolCallChunk, CreateChatCompletionStreamResponse};
    use futures::{StreamExt, stream};

    fn test_metrics() -> MetricsRecorder {
        let config = CallConfig::builder().build();
        MetricsRecorder::new(&config, "test")
    }

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

        let stream = stream::empty::<Result<CreateChatCompletionStreamResponse, async_openai::error::OpenAIError>>();
        let mut processed = ProcessedStream::new(stream, test_metrics());
        let response = processed.process_tool_call_chunk(chunk).unwrap();
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
        let mut processed = process_stream(stream, test_metrics());

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

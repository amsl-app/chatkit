use crate::error::{ChatKitError, StreamingError, ToolCallError};
use crate::messages::{AssistantMessage, TextContent, TokenUsage, ToolContent, extract_thinking, reject_empty};
use async_openai::config::Config;
use async_stream::try_stream;
use futures::{Stream, StreamExt};
use reqwest::header::{AUTHORIZATION, HeaderMap};
use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;
use std::fmt;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::Instant;
use typed_builder::TypedBuilder;

pub(crate) const ORGANIZATION_HEADER: &str = "OpenAI-Organization";
pub(crate) const PROJECT_HEADER: &str = "OpenAI-Project";

/// Request Types
pub(crate) const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";

pub(crate) const DEFAULT_MODEL: &str = "gpt-5.4-mini";

/// Connection and timeout settings for an LLM API endpoint.
#[derive(TypedBuilder, Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CallConfig {
    /// Covers the full request lifecycle including retries.
    #[builder(default = Duration::from_secs(30))]
    pub total_timeout: Duration,
    /// Per-attempt limit; for streaming, only applies to the initial connection.
    #[builder(default = Duration::from_secs(20))]
    pub iteration_timeout: Duration,
    #[builder(default = DEFAULT_API_BASE.into())]
    pub api_base: String,
    #[builder(default)]
    #[serde(skip)]
    pub api_key: SecretString,
    #[builder(default)]
    pub org_id: String,
    #[builder(default)]
    pub project_id: String,
    #[builder(default)]
    #[serde(skip)]
    pub custom_headers: HeaderMap,
}

impl Config for CallConfig {
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if !self.org_id.is_empty() {
            headers.insert(ORGANIZATION_HEADER, self.org_id.as_str().parse().unwrap());
        }

        if !self.project_id.is_empty() {
            headers.insert(PROJECT_HEADER, self.project_id.as_str().parse().unwrap());
        }

        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", self.api_key.expose_secret())
                .as_str()
                .parse()
                .unwrap(),
        );

        // Merge custom headers, with custom headers taking precedence
        for (key, value) in &self.custom_headers {
            headers.insert(key, value.clone());
        }

        headers
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.api_base, path)
    }

    fn query(&self) -> Vec<(&str, &str)> {
        vec![]
    }

    fn api_base(&self) -> &str {
        &self.api_base
    }

    fn api_key(&self) -> &SecretString {
        &self.api_key
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl From<ReasoningEffort> for async_openai::types::chat::ReasoningEffort {
    fn from(value: ReasoningEffort) -> Self {
        match value {
            ReasoningEffort::None => async_openai::types::chat::ReasoningEffort::None,
            ReasoningEffort::Minimal => async_openai::types::chat::ReasoningEffort::Minimal,
            ReasoningEffort::Low => async_openai::types::chat::ReasoningEffort::Low,
            ReasoningEffort::Medium => async_openai::types::chat::ReasoningEffort::Medium,
            ReasoningEffort::High => async_openai::types::chat::ReasoningEffort::High,
            ReasoningEffort::Xhigh => async_openai::types::chat::ReasoningEffort::Xhigh,
        }
    }
}

#[derive(TypedBuilder, Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CallOptions {
    #[builder(default)]
    pub streaming: bool,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[builder(default = DEFAULT_MODEL.into())]
    pub model: String,
}

/// Response Types

#[derive(Debug, Clone)]
pub enum ChatKitResponse {
    Message(AssistantMessage),
    Stream(StreamResponse),
}

impl ChatKitResponse {
    pub fn expect_message(self) -> Result<AssistantMessage, ChatKitError> {
        match self {
            ChatKitResponse::Message(msg) => Ok(msg),
            ChatKitResponse::Stream(_) => Err(ChatKitError::UnexpectedResponseFormat(
                "expected message response, got stream".into(),
            )),
        }
    }

    pub fn expect_stream(self) -> Result<StreamResponse, ChatKitError> {
        match self {
            ChatKitResponse::Stream(stream) => Ok(stream),
            ChatKitResponse::Message(_) => Err(ChatKitError::UnexpectedResponseFormat(
                "expected stream response, got message".into(),
            )),
        }
    }
}

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
    mut stream: impl Stream<
        Item = Result<async_openai::types::chat::CreateChatCompletionStreamResponse, async_openai::error::OpenAIError>,
    > + Unpin
    + Send
    + 'static,
    start_time: Instant,
    service: String,
    model: Cow<'static, str>,
) -> BoxedStream {
    let mut in_think_block = false;
    let mut buffer = String::new();

    try_stream! {
        let mut first_token_received = false;
        let mut previous_tool_call_id: Option<String> = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let tokens = chunk.usage.as_ref().map(|u| TokenUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            });

            let first = chunk.choices.into_iter().next();

            if let Some(first) = first {
                if !first_token_received
                    && (first.delta.content.is_some() || first.delta.tool_calls.is_some()) {
                        first_token_received = true;
                        #[cfg(feature = "metrics")]
                        {
                            // The precision loss is fine here, as we are only using it for metrics.
                            // TODO use as_millis_f64() once it is stable
                            #[allow(clippy::cast_precision_loss)]
                            metrics::histogram!(
                                "llm_time_to_first_token_ms",
                                "service" => service.clone(),
                                "model" => model.clone(),
                            ).record(start_time.elapsed().as_millis() as f64);
                        }
                    }
                if let Some(tool_calls) = first.delta.tool_calls {
                    let tool_calls: Vec<ToolContent> = tool_calls.into_iter().map(|tc| {
                        let result = process_tool_call_chunk(previous_tool_call_id.clone(), tc);
                        if let Ok(ref tool_call) = result {
                            previous_tool_call_id = Some(tool_call.id.clone());
                        }
                        result
                    }).collect()?;

                    yield AssistantMessage {
                        name: None,
                        text: None,
                        tools: tool_calls,
                        tokens,
                    }
                } else if let Some(refusal) = first.delta.refusal {
                    yield AssistantMessage {
                        name: None,
                        text: Some(TextContent { text: None, thinking: None, refusal: Some(refusal) }),
                        tools: Vec::new(),
                        tokens,
                    }
                } else if let Some(content) = first.delta.content {
                    buffer.push_str(&content);

                    loop {
                        if !in_think_block {
                            if let Some(pos) = buffer.find("<think>") {
                                let text = buffer[..pos].to_string();
                                buffer.drain(..pos + 7);
                                in_think_block = true;

                                if let Some(text) = reject_empty(text) {
                                    yield AssistantMessage {
                                        name: None,
                                        text: Some(TextContent { text: Some(text), thinking: None, refusal: None }),
                                        tools: Vec::new(),
                                        tokens: None,
                                    }
                                }
                            } else {
                                // No <think> tag found.
                                // We can yield everything up to the last '<' to avoid yielding a partial tag.
                                if let Some(last_lt) = buffer.rfind('<') {
                                    // Check if the content after '<' could be a start of "think>"
                                    let remaining = &buffer[last_lt..];
                                    if "<think>".starts_with(remaining) {
                                        let to_yield = buffer[..last_lt].to_string();
                                        buffer.drain(..last_lt);
                                        if let Some(text) = reject_empty(to_yield) {
                                             yield AssistantMessage {
                                                name: None,
                                                text: Some(TextContent { text: Some(text), thinking: None, refusal: None }),
                                                tools: Vec::new(),
                                                tokens: None,
                                            }
                                        }
                                        break; // Wait for next chunk
                                    }
                                }

                                let to_yield = buffer.clone();
                                buffer.clear();
                                if let Some(text) = reject_empty(to_yield) {
                                    yield AssistantMessage {
                                        name: None,
                                        text: Some(TextContent { text: Some(text), thinking: None, refusal: None }),
                                        tools: Vec::new(),
                                        tokens: None,
                                    }
                                }
                                break;
                            }
                        } else if let Some(pos) = buffer.find("</think>") {
                            let thinking = buffer[..pos].to_string();
                            buffer.drain(..pos + 8);
                            in_think_block = false;

                            let thinking = reject_empty(thinking);

                            yield AssistantMessage {
                                name: None,
                                text: Some(TextContent { text: None, thinking, refusal: None }),
                                tools: Vec::new(),
                                tokens: None,
                            }
                        } else {
                            // No </think> tag found.
                            // We can yield everything up to the last '<' to avoid yielding a partial tag.
                            if let Some(last_lt) = buffer.rfind('<') {
                                let remaining = &buffer[last_lt..];
                                if "</think>".starts_with(remaining) {
                                    let to_yield = buffer[..last_lt].to_string();
                                    buffer.drain(..last_lt);
                                    if let Some(thinking) = reject_empty(to_yield) {
                                         yield AssistantMessage {
                                            name: None,
                                            text: Some(TextContent { text: None, thinking: Some(thinking), refusal: None }),
                                            tools: Vec::new(),
                                            tokens: None,
                                        }
                                    }
                                    break;
                                }
                            }

                            let to_yield = buffer.clone();
                            buffer.clear();
                            if let Some(thinking) = reject_empty(to_yield) {
                                yield AssistantMessage {
                                    name: None,
                                    text: Some(TextContent { text: None, thinking: Some(thinking), refusal: None }),
                                    tools: Vec::new(),
                                    tokens: None,
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        #[cfg(feature = "metrics")]
        {
            // The precision loss is fine here, as we are only using it for metrics.
            // TODO use as_millis_f64() once it is stable
            #[allow(clippy::cast_precision_loss)]
            metrics::histogram!(
                "llm_time_to_last_token_ms",
                "service" => service,
                "model" => model.clone(),
            ).record(start_time.elapsed().as_millis() as f64);
        }
    }
    .boxed()
}

fn process_tool_call_chunk(
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
    use super::*;
    use async_openai::types::chat::{ChatCompletionMessageToolCallChunk, CreateChatCompletionStreamResponse};
    use futures::{StreamExt, stream};

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
}

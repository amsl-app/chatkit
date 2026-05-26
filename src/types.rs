use crate::error::ChatKitError;
use crate::messages::AssistantMessage;
use crate::stream::StreamResponse;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

pub(crate) const DEFAULT_MODEL: &str = "gpt-5.4-mini";

pub use crate::config::CallConfig;

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

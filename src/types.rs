use crate::error::ChatKitError;
use crate::messages::AssistantMessage;
use crate::stream::StreamResponse;
use async_openai::config::Config;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName};
use schemars::JsonSchema;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::time::Duration;
use typed_builder::TypedBuilder;

pub(crate) const ORGANIZATION_HEADER: HeaderName = HeaderName::from_static("openai-organization");
pub(crate) const PROJECT_HEADER: HeaderName = HeaderName::from_static("openai-project");

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
            insert_header(&mut headers, ORGANIZATION_HEADER, &self.org_id);
        }

        if !self.project_id.is_empty() {
            insert_header(&mut headers, PROJECT_HEADER, &self.project_id);
        }

        insert_header(
            &mut headers,
            AUTHORIZATION,
            &format!("Bearer {}", self.api_key.expose_secret()),
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

fn insert_header(headers: &mut HeaderMap, header: HeaderName, value: &str) {
    headers.insert(
        &header,
        value
            .parse()
            .inspect_err(|error| {
                tracing::error!(
                    header = header.as_str(),
                    value,
                    error = error as &dyn Error,
                    "invalid OpenAI header value"
                )
            })
            .unwrap(),
    );
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

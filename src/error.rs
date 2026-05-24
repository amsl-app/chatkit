use futures_retry_policies::ShouldRetry;
use std::borrow::Cow;
use std::error::Error;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ToolCallError {
    #[error("Syntax returned by LLM is invalid")]
    InvalidSyntax,

    #[error("No tool call in LLM response even though one was expected")]
    Missing,

    #[error("Tool call is missing an ID")]
    MissingToolId,

    #[error("Schema provided for tool call is invalid: {0}")]
    InvalidSchema(String),
}

#[derive(Debug, Error)]
pub enum ChatKitError {
    #[error(transparent)]
    Api(#[from] async_openai::error::OpenAIError),

    #[error(transparent)]
    Serde(#[from] serde_json::Error),

    #[error(transparent)]
    ToolCall(#[from] ToolCallError),

    #[error("No response from LLM")]
    EmptyResponse,

    #[error("Unexpected response format: {0}")]
    UnexpectedResponseFormat(Cow<'static, str>),

    #[error("Operation timed out")]
    Timeout,

    #[error(transparent)]
    HttpClientBuild(#[from] reqwest::Error),
}

impl ShouldRetry for ChatKitError {
    fn should_retry(&self, _: u32) -> bool {
        true
    }
}

pub type StreamingError = Box<dyn Error + Send + Sync>;

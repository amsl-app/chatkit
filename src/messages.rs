use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ChatKitError;
use crate::utils;

/// System Message

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SystemMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub content: String,
}

impl From<SystemMessage> for async_openai::types::chat::ChatCompletionRequestSystemMessage {
    fn from(msg: SystemMessage) -> Self {
        async_openai::types::chat::ChatCompletionRequestSystemMessage {
            name: msg.name,
            content: async_openai::types::chat::ChatCompletionRequestSystemMessageContent::Text(msg.content),
        }
    }
}

/// User Message
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UserMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub content: String,
}

impl From<UserMessage> for async_openai::types::chat::ChatCompletionRequestUserMessage {
    fn from(msg: UserMessage) -> Self {
        async_openai::types::chat::ChatCompletionRequestUserMessage {
            name: msg.name,
            content: async_openai::types::chat::ChatCompletionRequestUserMessageContent::Text(msg.content),
        }
    }
}

/// Assistant Message

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<TokenUsage>,
}

impl AssistantMessage {
    pub(crate) fn tools(tools: Vec<ToolContent>, tokens: Option<TokenUsage>) -> Self {
        Self {
            name: None,
            text: None,
            tools,
            tokens,
        }
    }

    pub(crate) fn refusal(refusal: String, tokens: Option<TokenUsage>) -> Self {
        Self::text_content(None, None, Some(refusal), tokens)
    }

    pub(crate) fn text(text: String) -> Self {
        Self::text_content(Some(text), None, None, None)
    }

    pub(crate) fn thinking(thinking: Option<String>) -> Self {
        Self::text_content(None, thinking, None, None)
    }

    pub(crate) fn text_content(
        text: Option<String>,
        thinking: Option<String>,
        refusal: Option<String>,
        tokens: Option<TokenUsage>,
    ) -> Self {
        Self {
            name: None,
            text: Some(TextContent {
                text,
                thinking,
                refusal,
            }),
            tools: Vec::new(),
            tokens,
        }
    }

    /// Extracts the `TextContent` from the current instance, if available.
    ///
    /// # Errors
    /// Returns a `ChatKitError::UnexpectedResponseFormat` when the `text` field is not present.
    ///
    /// # Example
    /// ```rust
    /// use chatkit::messages::{AssistantMessage, TextContent};
    /// # fn get_message() -> AssistantMessage {
    /// #     AssistantMessage {
    /// #         name: None,
    /// #         text: Some(TextContent {
    /// #                 text: Some("The answer is 42".to_string()),
    /// #                 thinking: None,
    /// #                 refusal: None,
    /// #             }),
    /// #         tools: Vec::new(),
    /// #         tokens: None,
    /// #     }
    /// # }
    /// # let message = get_message();
    /// let text = message.expect_text().unwrap();
    /// assert_eq!(text.text, Some("The answer is 42".to_string()));
    /// ```
    pub fn expect_text(self) -> Result<TextContent, ChatKitError> {
        self.text
            .ok_or_else(|| ChatKitError::UnexpectedResponseFormat("expected text content, got tool calls".into()))
    }

    pub fn expect_tools(self) -> Result<Vec<ToolContent>, ChatKitError> {
        if self.tools.is_empty() {
            Err(ChatKitError::UnexpectedResponseFormat(
                "expected tool calls, got text content".into(),
            ))
        } else {
            Ok(self.tools)
        }
    }
}

impl TryFrom<AssistantMessage> for async_openai::types::chat::ChatCompletionRequestAssistantMessage {
    type Error = ChatKitError;

    fn try_from(value: AssistantMessage) -> Result<Self, Self::Error> {
        let content: Option<async_openai::types::chat::ChatCompletionRequestAssistantMessageContent> =
            value.text.and_then(|t| {
                t.text
                    .map(async_openai::types::chat::ChatCompletionRequestAssistantMessageContent::Text)
            });

        let tool_calls: Option<Vec<async_openai::types::chat::ChatCompletionMessageToolCalls>> =
            if value.tools.is_empty() {
                None
            } else {
                Some(
                    value
                        .tools
                        .into_iter()
                        .map(TryInto::try_into)
                        .collect::<Result<_, ChatKitError>>()?,
                )
            };

        Ok(async_openai::types::chat::ChatCompletionRequestAssistantMessage {
            name: value.name,
            content,
            tool_calls,
            refusal: None,
            audio: None,
            #[allow(deprecated)]
            function_call: None,
        })
    }
}

impl TryFrom<async_openai::types::chat::CreateChatCompletionResponse> for AssistantMessage {
    type Error = ChatKitError;

    fn try_from(
        value: async_openai::types::chat::CreateChatCompletionResponse,
    ) -> Result<AssistantMessage, Self::Error> {
        let tokens = value.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        let first = value.choices.into_iter().next().ok_or(ChatKitError::EmptyResponse)?;

        if let Some(tool_calls) = first.message.tool_calls {
            let tool_calls: Vec<ToolContent> = tool_calls
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?;

            Ok(AssistantMessage::tools(tool_calls, tokens))
        } else if let Some(refusal) = first.message.refusal {
            Ok(AssistantMessage::refusal(refusal, tokens))
        } else if let Some(content) = first.message.content {
            let content_len = content.len();
            let (thinking, text) = utils::extract_thinking(content);

            let text = utils::reject_empty(text);

            let thinking_len = thinking.as_ref().map(String::len);
            let text_len = text.as_ref().map(String::len);

            tracing::debug!(
                content_len,
                has_thinking = thinking.is_some(),
                thinking_len,
                has_text = text.is_some(),
                text_len,
                "cleaned message content"
            );
            Ok(AssistantMessage::text_content(text, thinking, None, tokens))
        } else {
            Err(ChatKitError::EmptyResponse)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TextContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Extracted from `<think>…</think>` tags in the model output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolContent {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    pub arguments: Value,
}

impl TryFrom<ToolContent> for async_openai::types::chat::ChatCompletionMessageToolCall {
    type Error = ChatKitError;

    fn try_from(value: ToolContent) -> Result<Self, Self::Error> {
        Ok(async_openai::types::chat::ChatCompletionMessageToolCall {
            id: value.id,
            function: async_openai::types::chat::FunctionCall {
                name: value.name,
                arguments: serde_json::to_string(&value.arguments)?,
            },
        })
    }
}

impl TryFrom<async_openai::types::chat::ChatCompletionMessageToolCall> for ToolContent {
    type Error = ChatKitError;

    fn try_from(value: async_openai::types::chat::ChatCompletionMessageToolCall) -> Result<Self, Self::Error> {
        let async_openai::types::chat::ChatCompletionMessageToolCall { id, function } = value;
        let async_openai::types::chat::FunctionCall { name, arguments } = function;
        let (thinking, arguments) = utils::extract_thinking(arguments);
        let arguments_len = arguments.len();
        let arguments = Value::from_str(&arguments)?;
        let argument_keys: Vec<&str> = match &arguments {
            Value::Object(map) => map.keys().map(String::as_str).collect(),
            _ => Vec::new(),
        };
        tracing::debug!(
            arguments_len,
            argument_keys = ?argument_keys,
            "cleaned function call arguments"
        );

        Ok(ToolContent {
            id,
            name,
            thinking,
            arguments,
        })
    }
}

impl TryFrom<ToolContent> for async_openai::types::chat::ChatCompletionMessageToolCalls {
    type Error = ChatKitError;

    fn try_from(value: ToolContent) -> Result<Self, Self::Error> {
        let function = value.try_into()?;
        Ok(async_openai::types::chat::ChatCompletionMessageToolCalls::Function(
            function,
        ))
    }
}

impl TryFrom<async_openai::types::chat::ChatCompletionMessageToolCalls> for ToolContent {
    type Error = ChatKitError;

    fn try_from(value: async_openai::types::chat::ChatCompletionMessageToolCalls) -> Result<Self, Self::Error> {
        match value {
            async_openai::types::chat::ChatCompletionMessageToolCalls::Function(tool_call) => tool_call.try_into(),
            async_openai::types::chat::ChatCompletionMessageToolCalls::Custom(_) => Err(
                ChatKitError::UnexpectedResponseFormat("Custom tool calls are not supported".into()),
            ),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ToolMessage {
    pub id: String,
    pub content: String,
}

impl From<ToolMessage> for async_openai::types::chat::ChatCompletionRequestToolMessage {
    fn from(msg: ToolMessage) -> Self {
        async_openai::types::chat::ChatCompletionRequestToolMessage {
            tool_call_id: msg.id,
            content: async_openai::types::chat::ChatCompletionRequestToolMessageContent::Text(msg.content),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "role", content = "data")]
pub enum ChatKitMessage {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolMessage),
}

impl TryInto<async_openai::types::chat::ChatCompletionRequestMessage> for ChatKitMessage {
    type Error = ChatKitError;

    fn try_into(self) -> Result<async_openai::types::chat::ChatCompletionRequestMessage, Self::Error> {
        match self {
            ChatKitMessage::System(msg) => Ok(async_openai::types::chat::ChatCompletionRequestMessage::System(
                msg.into(),
            )),
            ChatKitMessage::User(msg) => Ok(async_openai::types::chat::ChatCompletionRequestMessage::User(
                msg.into(),
            )),
            ChatKitMessage::Assistant(msg) => Ok(async_openai::types::chat::ChatCompletionRequestMessage::Assistant(
                msg.try_into()?,
            )),
            ChatKitMessage::Tool(msg) => Ok(async_openai::types::chat::ChatCompletionRequestMessage::Tool(
                msg.into(),
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Memory(pub Vec<ChatKitMessage>);

impl TryFrom<Memory> for Vec<async_openai::types::chat::ChatCompletionRequestMessage> {
    type Error = ChatKitError;

    fn try_from(value: Memory) -> Result<Self, Self::Error> {
        value
            .0
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<_, ChatKitError>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{ChatCompletionMessageToolCall, CreateChatCompletionResponse};

    #[test]
    fn test_non_streaming_no_tools() {
        let json = r#"{
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>thinking hard</think>The answer is 42"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }"#;
        let response: CreateChatCompletionResponse = serde_json::from_str(json).unwrap();

        let message: AssistantMessage = response.try_into().unwrap();
        let text = message.text.unwrap();
        assert_eq!(text.thinking, Some("thinking hard".to_string()));
        assert_eq!(text.text, Some("The answer is 42".to_string()));
    }

    #[test]
    fn test_non_streaming_with_tools() {
        let json = r#"{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "test_tool",
                "arguments": "{\"arg\": \"<think>parsing json</think>value\"}"
            }
        }"#;
        let tool_call: ChatCompletionMessageToolCall = serde_json::from_str(json).unwrap();

        let response: ToolContent = tool_call.try_into().unwrap();
        assert_eq!(response.name, "test_tool");
        assert_eq!(response.thinking, Some("parsing json".to_string()));
        assert_eq!(response.arguments["arg"], "value");
    }
}

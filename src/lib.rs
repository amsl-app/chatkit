use crate::error::{ChatKitError, ToolCallError};
use crate::messages::{AssistantMessage, Memory, TokenUsage, ToolContent};
use crate::tools::{ToolChoice, ToolSchema};
use crate::types::{CallConfig, CallOptions, ChatKitResponse};

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::error::Error;
use stream::{StreamResponse, process_stream};
use tokio::time::Instant;
use tracing::instrument;

pub mod error;
pub mod messages;
mod stream;
pub mod tools;
pub mod types;
mod utils;

/// Makes a single LLM call, returning either a complete message or a stream depending on `options.streaming`.
#[instrument(skip(config))]
pub async fn llm_call(
    config: CallConfig,
    options: CallOptions,
    memory: Memory,
    tools: Vec<ToolSchema>,
    tool_choice: Option<ToolChoice>,
) -> Result<ChatKitResponse, ChatKitError> {
    let CallOptions {
        model,
        temperature,
        reasoning_effort,
        streaming,
    } = options;

    let start_time = Instant::now();

    let service = config.api_base.clone().into();

    let mut request = async_openai::types::chat::CreateChatCompletionRequestArgs::default();
    let messages: Vec<async_openai::types::chat::ChatCompletionRequestMessage> = memory.try_into()?;
    request.model(&model).messages(messages);

    if let Some(temperature) = temperature {
        request.temperature(temperature);
    }

    if let Some(reasoning_effort) = reasoning_effort {
        request.reasoning_effort(reasoning_effort);
    }

    let tools: Vec<_> = tools.into_iter().map(TryInto::try_into).collect::<Result<_, _>>()?;

    if !tools.is_empty() {
        tracing::debug!(tool_count = tools.len(), "adding tools to LLM request");
        request.tools(tools);

        if let Some(tool_choice) = tool_choice {
            request.tool_choice(tool_choice);
        }
    }

    tracing::debug!(?request, "built LLM request");

    let request = request.build()?;

    let mut http_client_builder = reqwest::Client::builder();
    if streaming {
        // For streaming, only set a connect timeout — a full response timeout would kill
        // long-running streams before they complete. Also, disable auto-decompression since
        // SSE streams cannot be gzip-decoded incrementally.
        http_client_builder = http_client_builder
            .connect_timeout(config.iteration_timeout)
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd();
    } else {
        http_client_builder = http_client_builder.timeout(config.iteration_timeout);
    }

    let http_client = http_client_builder.build().map_err(|error| {
        tracing::error!(error = &error as &dyn Error, "failed to build http client for llm call");
        ChatKitError::HttpClientBuild(error)
    })?;

    let http_service = tower::ServiceBuilder::new()
        .concurrency_limit(8)
        .timeout(config.total_timeout)
        .layer(async_openai::middleware::retry::OpenAIRetryLayer::default())
        .service(async_openai::middleware::ReqwestService::new(http_client));

    let client = async_openai::Client::with_config(config).with_http_service(http_service);

    if streaming {
        tracing::debug!("using streaming LLM call");
        let res = client.chat().create_stream(request).await;
        match res {
            Ok(stream) => {
                let stream = process_stream(stream, start_time, service, model.into());
                Ok(ChatKitResponse::Stream(StreamResponse::new(stream)))
            }
            Err(error) => Err(ChatKitError::Api(error)),
        }
    } else {
        let res: Result<async_openai::types::chat::CreateChatCompletionResponse, async_openai::error::OpenAIError> =
            client.chat().create(request).await;

        #[cfg(feature = "metrics")]
        {
            // The precision loss is fine here, as we are only using it for metrics.
            // TODO use as_millis_f64() once it is stable
            #[allow(clippy::cast_precision_loss)]
            let elapsed = start_time.elapsed().as_millis() as f64;
            metrics::histogram!(
                "llm_time_to_last_token_ms",
                "service" => service,
                "model" => model,
            )
            .record(elapsed);
        }

        tracing::debug!(?res, "received LLM response");
        let chat_completion = res.inspect_err(|error| {
            tracing::warn!(error = error as &dyn Error, "LLM call failed");
        })?;

        let message: AssistantMessage = chat_completion.try_into()?;
        Ok(ChatKitResponse::Message(message))
    }
}

/// Forces the LLM to return structured output of type `T` by registering it as a required tool call.
#[instrument(skip(config), err)]
pub async fn llm_single_tool_call<T: DeserializeOwned + JsonSchema>(
    config: CallConfig,
    options: CallOptions,
    memory: Memory,
) -> Result<(T, Option<TokenUsage>), ChatKitError> {
    let schema = schemars::schema_for!(T);
    let tool_schema: ToolSchema = schema.into();
    let tool_name = tool_schema
        .name()
        .ok_or(ChatKitError::ToolCall(ToolCallError::InvalidSchema(
            "Missing title in schema".to_string(),
        )))?
        .to_string();

    let llm_response = llm_call(
        config,
        options,
        memory,
        vec![tool_schema],
        Some(ToolChoice::Named(tool_name.clone())),
    )
    .await?;

    let ChatKitResponse::Message(llm_response) = llm_response else {
        return Err(ChatKitError::UnexpectedResponseFormat(
            "Expected non-streaming response for single tool call".into(),
        ));
    };

    let tool_response: Option<ToolContent> = llm_response.tools.into_iter().find(|t| t.name == tool_name);

    let tool_response = tool_response.ok_or_else(|| ChatKitError::ToolCall(ToolCallError::Missing))?;

    let res: T = serde_json::from_value(tool_response.arguments)?;
    Ok((res, llm_response.tokens))
}

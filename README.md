# llmkit

Thin async Rust wrapper around any OpenAI-compatible chat API with streaming, tool calling, and structured output support.

## Features

- Non-streaming and streaming responses
- Structured output via tool calling (`llm_single_tool_call`)
- Automatic `<think>…</think>` extraction for reasoning models
- Manual or derive-based tool schemas
- Optional `metrics` feature (histograms for TTFT / TTLT)

## Quick start

```toml
[dependencies]
llmkit = { path = "." }
tokio = { version = "1", features = ["macros", "rt-current-thread"] }
```

### Simple chat

```rust
use llmkit::{llm_call, messages::{LLMKitMessage, Memory, SystemMessage, UserMessage}, types::{CallConfig, CallOptions}};

let config = CallConfig::builder().api_key(api_key.into()).build();
let options = CallOptions::builder().model("gpt-4o-mini".into()).build();
let memory = Memory(vec![
    LLMKitMessage::User(UserMessage { name: None, content: "Hello!".into() }),
]);

let msg = llm_call(config, options, memory, vec![], None)
    .await?
    .expect_message()?;
println!("{}", msg.text.unwrap().text.unwrap_or_default());
```

### Structured output

```rust
use llmkit::llm_single_tool_call;
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, JsonSchema, Deserialize)]
#[schemars(description = "Sentiment analysis result")]
struct Sentiment {
    /// positive, negative, or neutral
    sentiment: String,
    /// 0.0–1.0
    confidence: f32,
}

let (result, _tokens) = llm_single_tool_call::<Sentiment>(config, options, memory).await?;
println!("{} ({:.0}%)", result.sentiment, result.confidence * 100.0);
```

## Core types

| Type | Description |
|---|---|
| `CallConfig` | API endpoint, key, and timeouts (builder pattern) |
| `CallOptions` | Model, temperature, streaming, reasoning effort |
| `Memory` | Ordered list of `LLMKitMessage`s (conversation history) |
| `LLMKitResponse` | Either `Message(AssistantMessage)` or `Stream(StreamResponse)` |
| `ToolSchema` | JSON Schema → OpenAI function tool (via `schemars` or `OpenApiField`) |

## Examples

```sh
OPENAI_API_KEY=sk-... cargo run --example simple_chat
OPENAI_API_KEY=sk-... cargo run --example structured_output
OPENAI_API_KEY=sk-... cargo run --example streaming
```

## Features

```toml
llmkit = { path = ".", features = ["metrics"] }
```

Enables `metrics` histograms: `llm_time_to_first_token_ms` and `llm_time_to_last_token_ms`, tagged with `service` and `model`.

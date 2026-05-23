use llmkit::{
    llm_single_tool_call,
    messages::{LLMKitMessage, Memory, SystemMessage, UserMessage},
    types::{CallConfig, CallOptions},
};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, JsonSchema, Deserialize)]
#[schemars(description = "Sentiment analysis result for a piece of text")]
struct SentimentResult {
    /// The detected sentiment: positive, negative, or neutral
    sentiment: String,
    /// Confidence score between 0.0 and 1.0
    confidence: f32,
    /// Brief explanation of the detected sentiment
    explanation: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let config = CallConfig::builder().api_key(api_key.into()).build();

    let options = CallOptions::builder().model("gpt-4o-mini".into()).build();

    let text = "I absolutely loved the movie! The acting was superb and the story kept me engaged throughout.";

    let memory = Memory(vec![
        LLMKitMessage::System(SystemMessage {
            name: None,
            content: "You are a sentiment analysis assistant.".into(),
        }),
        LLMKitMessage::User(UserMessage {
            name: None,
            content: format!("Analyze the sentiment of this text: {text}"),
        }),
    ]);

    let (result, tokens) = llm_single_tool_call::<SentimentResult>(config, options, memory)
        .await
        .expect("LLM call failed");

    println!("Sentiment:   {}", result.sentiment);
    println!("Confidence:  {:.2}", result.confidence);
    println!("Explanation: {}", result.explanation);

    if let Some(tokens) = tokens {
        println!("\nTokens used: {}", tokens.total_tokens);
    }
}

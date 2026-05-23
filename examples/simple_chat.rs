use llmkit::{
    llm_call,
    messages::{LLMKitMessage, Memory, SystemMessage, UserMessage},
    types::{CallConfig, CallOptions},
};

#[tokio::main(flavor = "current_thread")]
async fn main() {
     dotenvy::dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let config = CallConfig::builder().api_key(api_key.into()).build();

    let options = CallOptions::builder().model("gpt-4o-mini".into()).build();

    let memory = Memory(vec![
        LLMKitMessage::System(SystemMessage {
            name: None,
            content: "You are a concise assistant.".into(),
        }),
        LLMKitMessage::User(UserMessage {
            name: None,
            content: "What is the capital of France?".into(),
        }),
    ]);

    let msg = llm_call(config, options, memory, vec![], None)
        .await
        .expect("LLM call failed")
        .expect_message()
        .expect("expected message response");

    if let Some(text) = msg.text {
        println!("{}", text.text.unwrap_or_default());
    }
}

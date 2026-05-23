use chatkit::{
    llm_call,
    messages::{ChatKitMessage, Memory, UserMessage},
    types::{CallConfig, CallOptions},
};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let config = CallConfig::builder().api_key(api_key.into()).build();

    let options = CallOptions::builder()
        .model("gpt-4o-mini".into())
        .streaming(true)
        .build();

    let memory = Memory(vec![ChatKitMessage::User(UserMessage {
        name: None,
        content: "Count from 1 to 5, one number per line.".into(),
    })]);

    let stream = llm_call(config, options, memory, vec![], None)
        .await
        .expect("LLM call failed")
        .expect_stream()
        .expect("expected stream response");

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("Stream error");
        if let Some(text) = chunk.text {
            if let Some(content) = text.text {
                print!("{content}");
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
        }
    }
    println!();
}

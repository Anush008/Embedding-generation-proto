use crate::github::Repository;
use openai_api_rs::v1::{api::Client, chat_completion::ChatCompletionMessage};
use serde::Deserialize;
use std::env;

#[derive(Deserialize)]
pub struct Query {
    pub repository: Repository,
    pub query: String,
}

pub struct Conversation {
    pub query: Query,
    pub client: Client,
    pub messages: Vec<ChatCompletionMessage>
}

impl Conversation {
    pub fn new(query: Query) -> Self {
        Self {
            query,
            messages: Vec::new(),
            client: Client::new(env::var("OPENAI_API_KEY").unwrap().to_string()),
        }
    }
}

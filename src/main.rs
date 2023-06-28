mod utils;
use std::path::Path;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let embeddings = utils::embeddings::Embeddings::initialize(Path::new("model")).expect("Failed to initialize model");
    println!("Model loaded");
    let emb = embeddings
        .embed_repo("open-sauced", "hot", "beta")
        .await
        .unwrap();
    println!("{} embeddings generated", emb.len());
    let emb = embeddings
        .embed_repo("open-sauced", "ai", "beta")
        .await
        .unwrap();
    println!("{} embeddings generated", emb.len());
    let emb = embeddings
        .embed_repo("open-sauced", "insights", "beta")
        .await
        .unwrap();
    println!("{} embeddings generated", emb.len());
    Ok(())
}

use crate::github::RepositoryEmbeddings;
use crate::prelude::*;
mod qdrant;
use async_trait::async_trait;

pub use qdrant::*;

#[async_trait]
pub trait RepositoryEmbeddingsDB {
    async fn insert_repo_embeddings(&self, repo: RepositoryEmbeddings) -> Result<()>;
}

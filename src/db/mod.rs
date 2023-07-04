use crate::embeddings::Embeddings;
use crate::github::{RepositoryEmbeddings, File};
use crate::prelude::*;
mod qdrant;
use async_trait::async_trait;

pub use qdrant::*;

#[async_trait]
pub trait RepositoryEmbeddingsDB {
    async fn insert_repo_embeddings(&self, repo: RepositoryEmbeddings) -> Result<()>;
    
    async fn get_relevant_files(
        &self,
        repo_owner: &str,
        repo_name: &str,
        repo_branch: &str,
        query_embeddings: Embeddings,
        limit: u64,
    ) -> Result<Vec<File>>;
}

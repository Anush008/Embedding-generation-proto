use std::collections::HashMap;

use super::RepositoryEmbeddingsDB;
use crate::{
    embeddings::Embeddings,
    github::{fetch_file_content, File, FileEmbeddings, RepositoryEmbeddings},
    prelude::*,
};
use anyhow::Ok;
use async_trait::async_trait;
use qdrant_client::{
    prelude::*,
    qdrant::{vectors_config::Config, VectorParams, VectorsConfig},
};
use rayon::prelude::*;
use uuid::Uuid;

pub struct QdrantDB {
    client: QdrantClient,
}

#[async_trait]
impl RepositoryEmbeddingsDB for QdrantDB {
    async fn insert_repo_embeddings(&self, repo: RepositoryEmbeddings) -> Result<()> {
        self.client
            .create_collection(&CreateCollection {
                collection_name: repo.repo_id.clone(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: 384,
                        distance: Distance::Cosine.into(),
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await?;

        let points: Vec<PointStruct> = repo
            .file_embeddings
            .into_par_iter()
            .map(|file| {
                let FileEmbeddings { path, embeddings } = file;
                let payload: Payload = HashMap::from([("path", Value::from(path))]).into();

                PointStruct::new(Uuid::new_v4().to_string(), embeddings, payload)
            })
            .collect();
        self.client
            .upsert_points(repo.repo_id, points, None)
            .await?;
        Ok(())
    }

    async fn get_relevant_files(
        &self,
        repo_owner: &str,
        repo_name: &str,
        repo_branch: &str,
        query_embeddings: Embeddings,
        limit: u64,
    ) -> Result<Vec<File>> {
        let search_response = self
            .client
            .search_points(&SearchPoints {
                collection_name: format!("{repo_owner}-{repo_branch}-{repo_name}"),
                vector: query_embeddings,
                with_payload: Some(true.into()),
                limit,
                ..Default::default()
            })
            .await?;
        let futures: Vec<_> = search_response
            .result
            .into_iter()
            .map(|point| {
                let path = point.payload["path"].to_string();
                async {
                    let content =
                        fetch_file_content(repo_owner, repo_name, repo_branch, &path)
                            .await
                            .unwrap_or_default();
                    let length = content.len();
                    File {
                        path,
                        content,
                        length,
                    }
                }
            })
            .collect();
        let files: Vec<File> = futures::future::join_all(futures).await;
        Ok(files)
    }
}
impl QdrantDB {
    pub fn initialize() -> Result<QdrantDB> {
        let mut config = QdrantClientConfig::from_url(
            &std::env::var("QDRANT_URL").expect("QDRANT_URL environment variable not set"),
        );
        config.set_api_key(
            &std::env::var("QDRANT_API_KEY").expect("QDRANT_API_KEY environment variable not set"),
        );
        let client = QdrantClient::new(Some(config))?;
        Ok(QdrantDB { client })
    }
}

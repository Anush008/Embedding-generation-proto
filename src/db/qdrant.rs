use std::collections::HashMap;

use super::RepositoryEmbeddingsDB;
use crate::{
    github::{FileEmbeddings, RepositoryEmbeddings},
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
        let collection_id = repo.repo_id.replace("/", "-");
        self.client
            .create_collection(&CreateCollection {
                collection_name: collection_id.clone(),
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
            .upsert_points(collection_id, points, None)
            .await?;
        Ok(())
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

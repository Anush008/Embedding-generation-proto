use crate::db::RepositoryEmbeddingsDB;
use actix_web::{
    post,
    web::{self, Json},
    HttpResponse, Responder,
};
use reqwest::StatusCode;
use serde::Deserialize;
use std::sync::Arc;

use crate::{db::QdrantDB, embeddings::Onnx, github::embed_repo};

#[derive(Deserialize)]
struct Data {
    repo_name: String,
    repo_owner: String,
    repo_branch: String,
}

#[post("/embeddings")]
async fn embeddings(
    data: Json<Data>,
    db: web::Data<Arc<QdrantDB>>,
    model: web::Data<Arc<Onnx>>,
) -> impl Responder {
    let Data {
        repo_name,
        repo_owner,
        repo_branch,
    } = data.into_inner();

    let embeddings = embed_repo(
        &repo_owner,
        &repo_name,
        &repo_branch,
        model.get_ref().as_ref(),
    )
    .await
    .unwrap();
    db.get_ref()
        .as_ref()
        .insert_repo_embeddings(embeddings)
        .await
        .unwrap();
    HttpResponse::new(StatusCode::CREATED)
}

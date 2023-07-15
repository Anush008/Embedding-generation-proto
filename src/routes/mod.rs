use crate::embeddings::EmbeddingsModel;
use crate::{db::RepositoryEmbeddingsDB, github::Repository};
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
struct Query {
    repo_owner: String,
    repo_name: String,
    repo_branch: String,
    query: String,
}

#[post("/embeddings")]
async fn embeddings(
    data: Json<Repository>,
    db: web::Data<Arc<QdrantDB>>,
    model: web::Data<Arc<Onnx>>,
) -> impl Responder {
    let embeddings = embed_repo(data.into_inner(), model.get_ref().as_ref())
        .await
        .unwrap();
    
    match db.get_ref()
        .as_ref()
        .insert_repo_embeddings(embeddings)
        .await {
            Ok(_) => HttpResponse::new(StatusCode::CREATED),
            Err(e) => {
                println!("Error inserting embeddings: {:?}", e);
                return HttpResponse::new(StatusCode::INTERNAL_SERVER_ERROR);
            }
        }
    
}

#[post("/query")]
async fn query(
    data: Json<Query>,
    db: web::Data<Arc<QdrantDB>>,
    model: web::Data<Arc<Onnx>>,
) -> impl Responder {
    let Query {
        repo_owner,
        repo_name,
        repo_branch,
        query,
    } = data.into_inner();

    let relevant_file_paths = db
        .get_ref()
        .as_ref()
        .get_relevant_files(
            Repository {
                owner: repo_owner,
                name: repo_name,
                branch: repo_branch,
            },
            model.get_ref().as_ref().embed(&query).unwrap(),
            2,
        )
        .await
        .unwrap();

    HttpResponse::Ok().json(relevant_file_paths)
}

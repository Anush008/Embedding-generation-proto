use actix_web::{post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use ndarray::Axis;
use ort::{
    tensor::{FromArray, InputTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, SessionBuilder,
};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};

pub struct Semantic {
    tokenizer: Arc<tokenizers::Tokenizer>,
    session: Arc<ort::Session>,
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

impl Semantic {
    pub fn initialize(model_dir: &Path) -> Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("Encode")
                .with_log_level(LoggingLevel::Warning)
                .with_execution_providers([ExecutionProvider::cpu()])
                .build()?,
        );

        let threads = 8;

        Ok(Self {
            tokenizer: tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .unwrap()
                .into(),
            session: SessionBuilder::new(&environment)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(threads)?
                .with_model_from_file(model_dir.join("model.onnx"))?
                .into(),
        })
    }

    pub fn embed(&self, sequence: &str) -> Result<Vec<f32>> {
        let tokenizer_output: tokenizers::Encoding = self.tokenizer.encode(sequence, true).unwrap();

        let input_ids = tokenizer_output.get_ids();
        let attention_mask = tokenizer_output.get_attention_mask();
        let token_type_ids = tokenizer_output.get_type_ids();
        let length = input_ids.len();

        let inputs_ids_array = ndarray::Array::from_shape_vec(
            (1, length),
            input_ids.iter().map(|&x| x as i64).collect(),
        )?;

        let attention_mask_array = ndarray::Array::from_shape_vec(
            (1, length),
            attention_mask.iter().map(|&x| x as i64).collect(),
        )?;

        let token_type_ids_array = ndarray::Array::from_shape_vec(
            (1, length),
            token_type_ids.iter().map(|&x| x as i64).collect(),
        )?;

        let outputs = self.session.run([
            InputTensor::from_array(inputs_ids_array.into_dyn()),
            InputTensor::from_array(attention_mask_array.into_dyn()),
            InputTensor::from_array(token_type_ids_array.into_dyn()),
        ])?;

        let output_tensor = outputs[0].try_extract().unwrap();
        let sequence_embedding = &*output_tensor.view();
        let pooled = sequence_embedding.mean_axis(Axis(1)).unwrap();
        Ok(pooled.to_owned().as_slice().unwrap().to_vec())
    }
}

#[derive(Deserialize)]
struct Data {
    string: String,
}

#[derive(Serialize)]
struct Embeddings {
    embeddings: Vec<f32>,
}

#[post("/embeddings")]
async fn embeddings(
    request: HttpRequest,
    data: web::Json<Data>,
) -> actix_web::Result<impl Responder> {
    let model = request
        .app_data::<Arc<Semantic>>()
        .expect("Pool app_data failed to load!");
    Ok(web::Json(model.embed(&data.string).unwrap()))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let semantic = Arc::new(Semantic::initialize(Path::new("model")).unwrap());

    println!("Model pool created!");

    HttpServer::new(move || {
        //Increase JSON payload size
        App::new()
            .app_data(web::JsonConfig::default().limit(8929566200))
            .app_data(semantic.clone())
            .route("/", web::get().to(HttpResponse::Ok))
            .service(embeddings)
    })
    .bind(("127.0.0.1", 3001))?
    .run()
    .await
}

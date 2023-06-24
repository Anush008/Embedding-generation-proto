use anyhow::Result;
use ort::{
    tensor::{FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, SessionBuilder,
};
use std::{sync::Arc, path::Path};
use ndarray::Axis;
use actix_web::{post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

pub struct Semantic {
    tokenizer: Arc<tokenizers::Tokenizer>,
    session: Arc<ort::Session>,
}

impl Semantic {
    pub fn initialize(
        model_dir: &Path,
    ) -> Result<Self> {
       let environment = Arc::new(
            Environment::builder()
                .with_name("Encode")
                .with_log_level(LoggingLevel::Warning)
                .with_execution_providers([ExecutionProvider::cpu()])
                .build()?,
        );

        let threads = num_cpus::get() as i16;
        dbg!(threads);

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

    pub fn embed(&self, sequence: &str) -> anyhow::Result<Vec<f32>> {
        let tokenizer_output: tokenizers::Encoding = self.tokenizer.encode(sequence, true).unwrap();

        let input_ids: &[u32] = tokenizer_output.get_ids();
        let attention_mask: &[u32] = tokenizer_output.get_attention_mask();
        let token_type_ids: &[u32] = tokenizer_output.get_type_ids();
        let length: usize = input_ids.len();

        let inputs_ids_array: ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>> = ndarray::Array::from_shape_vec(
            (1, length),
            input_ids.iter().map(|&x| x as i64).collect(),
        )?;

        let attention_mask_array: ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>> = ndarray::Array::from_shape_vec(
            (1, length),
            attention_mask.iter().map(|&x| x as i64).collect(),
        )?;

        let token_type_ids_array: ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>> = ndarray::Array::from_shape_vec(
            (1, length),
            token_type_ids.iter().map(|&x| x as i64).collect(),
        )?;

        let outputs: Vec<ort::tensor::DynOrtTensor<'_, ndarray::Dim<ndarray::IxDynImpl>>> = self.session.run([
            InputTensor::from_array(inputs_ids_array.into_dyn()),
            InputTensor::from_array(attention_mask_array.into_dyn()),
            InputTensor::from_array(token_type_ids_array.into_dyn()),
        ])?;

        let output_tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
        let sequence_embedding: &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<ndarray::IxDynImpl>> = &*output_tensor.view();
        let pooled: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> = sequence_embedding.mean_axis(Axis(1)).unwrap();
        Ok(pooled.to_owned().as_slice().unwrap().to_vec())
    }


}

#[derive(Deserialize)]
struct Data {
    strings: Vec<String>,
}

#[derive(Serialize)]
struct Embeddings {
    embeddings: Vec<Vec<f32>>,
}

#[post("/embeddings")]
async fn embeddings(request: HttpRequest, data: web::Json<Data>) -> actix_web::Result<impl Responder> {
    let model = request
        .app_data::<Arc<Semantic>>()
        .expect("Pool app_data failed to load!");
    dbg!(data.strings.len());
    let embeds: Vec<Vec<f32>> = data.strings.par_iter().map(|string| model.embed(string).unwrap()).collect();
    let embeddings = Embeddings { embeddings: embeds };
    Ok(web::Json(embeddings))
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
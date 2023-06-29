use ndarray::Axis;
use ort::{
    tensor::{FromArray, InputTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder,
};
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::*;
use std::{io::Read, thread::available_parallelism, time::Instant};
use std::{path::Path, sync::Arc};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub struct Embeddings {
    tokenizer: Arc<tokenizers::Tokenizer>,
    session: Arc<ort::Session>,
}

impl Embeddings {
    pub fn initialize(model_dir: &Path) -> Result<Self> {
        let environment = Arc::new(
            Environment::builder()
                .with_name("Embeddings")
                .with_execution_providers([ExecutionProvider::cpu()])
                .build()?,
        );

        let threads = available_parallelism().unwrap().get() as i16;

        Ok(Self {
            tokenizer: tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .unwrap()
                .into(),
            session: SessionBuilder::new(&environment)?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(threads)?
                .with_model_from_file(model_dir.join("model_quantized.onnx"))?
                .into(),
        })
    }

    fn embed(&self, sequence: &str) -> Result<Vec<f32>> {
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

    pub fn embed_strings(&self, strings: &Vec<String>) -> Vec<Vec<f32>> {
        strings
            .into_par_iter()
            .map(|string| self.embed(&string).unwrap())
            .collect()
    }

    pub async fn embed_repo(
        &self,
        repo_owner: &str,
        repo_name: &str,
        repo_branch: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let url = format!("https://github.com/{repo_owner}/{repo_name}/archive/{repo_branch}.zip");
        println!("\n\nDownloading repo from: {}\n", url);
        let time = Instant::now();
        let response = reqwest::get(url).await?.bytes().await?;
        let reader = std::io::Cursor::new(response);
        let mut archive = zip::ZipArchive::new(reader)?;
        let mut contents = Vec::new();
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let mut content = String::new();
            if file.is_file() {
                //Fails for non UTF-8 files
                match file.read_to_string(&mut content) {
                    Ok(_) => contents.push(content),
                    Err(_) => continue,
                };
            }
        }
        println!("Downloading repo took: {:?}\n", time.elapsed());
        println!("Generating embeddings for {} files", contents.len());
        let embedding = self.embed_strings(&contents);
        println!("Embedding took: {:?}\n", time.elapsed());
        Ok(embedding)
    }
}

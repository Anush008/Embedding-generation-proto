use crate::{
    embeddings::{Embeddings, EmbeddingsModel},
    prelude::*,
};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::io::Read;

#[derive(Debug, Default)]
pub struct File {
    path: String,
    content: String,
    size: String,
}

#[derive(Debug, Clone)]
pub struct FileEmbeddings {
    pub path: String,
    pub embeddings: Embeddings,
}
pub struct RepositoryEmbeddings {
    pub repo_id: String,
    pub file_embeddings: Vec<FileEmbeddings>,
}

pub async fn embed_repo<M: EmbeddingsModel + Send + Sync>(
    repo_owner: &str,
    repo_name: &str,
    repo_branch: &str,
    model: &M,
) -> Result<RepositoryEmbeddings> {
    let time = std::time::Instant::now();
    let files: Vec<File> = fetch_repo_files(repo_owner, repo_name, repo_branch).await?;
    println!("Time to fetch files: {:?}", time.elapsed());
    let time = std::time::Instant::now();
    let file_embeddings: Vec<FileEmbeddings> = files
        .into_par_iter()
        .filter_map(|file| {
            let File {
                path,
                size,
                content,
            } = file;
            let embed_content = format!(
                "File path: {}\nFile size: {} bytes\nFile content: {}",
                &path, &size, &content
            );
            let embeddings = model.embed(&embed_content).unwrap();
            Some(FileEmbeddings { path, embeddings })
        })
        .collect();
    println!("Time to embed files: {:?}", time.elapsed());

    Ok(RepositoryEmbeddings {
        repo_id: format!("{repo_owner}/{repo_name}"),
        file_embeddings,
    })
}

async fn fetch_repo_files(
    repo_owner: &str,
    repo_name: &str,
    repo_branch: &str,
) -> Result<Vec<File>> {
    let url = format!("https://github.com/{repo_owner}/{repo_name}/archive/{repo_branch}.zip");
    let response = reqwest::get(url).await?.bytes().await?;
    let reader = std::io::Cursor::new(response);
    let mut archive = zip::ZipArchive::new(reader)?;
    let files: Vec<File> = (0..archive.len())
        .filter_map(|file| {
            let mut file = archive.by_index(file).unwrap();
            if file.is_file() {
                let mut content = String::new();
                //Fails for non UTF-8 files
                match file.read_to_string(&mut content) {
                    Ok(_) => Some(File {
                        path: file.name().to_string(),
                        content: content,
                        size: file.size().to_string(),
                    }),
                    Err(_) => None,
                }
            } else {
                None
            }
        })
        .collect();
    Ok(files)
}
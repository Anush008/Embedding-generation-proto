use crate::{
    embeddings::{Embeddings, EmbeddingsModel},
    prelude::*,
};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    io::Read,
};

#[derive(Debug, Default, Serialize)]
pub struct File {
    pub path: String,
    pub content: String,
    pub length: usize,
}

impl Display for File {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "File path: {}\nFile length: {} bytes\nFile content: {}",
            &self.path, &self.length, &self.content
        )
    }
}

#[derive(Debug, Clone)]
pub struct FileEmbeddings {
    pub path: String,
    pub embeddings: Embeddings,
}

#[derive(Debug)]
pub struct RepositoryEmbeddings {
    pub repo_id: String,
    pub file_embeddings: Vec<FileEmbeddings>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Repository {
    pub owner: String,
    pub name: String,
    pub branch: String,
}

impl Display for Repository {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}-{}", &self.owner, &self.name, &self.branch)
    }
}
pub async fn embed_repo<M: EmbeddingsModel + Send + Sync>(
    repository: Repository,
    model: &M,
) -> Result<RepositoryEmbeddings> {
    let time = std::time::Instant::now();
    let files: Vec<File> = fetch_repo_files(repository.clone()).await?;
    println!("Time to fetch files: {:?}", time.elapsed());
    let time = std::time::Instant::now();
    let file_embeddings: Vec<FileEmbeddings> = files
        .into_par_iter()
        .filter_map(|file| {
            let embed_content = format!("{file}");
            let embeddings = model.embed(&embed_content).unwrap();
            Some(FileEmbeddings {
                path: file.path,
                embeddings,
            })
        })
        .collect();
    println!("Time to embed files: {:?}", time.elapsed());
    println!("{repository}");
    Ok(RepositoryEmbeddings {
        repo_id: format!("{repository}"),
        file_embeddings,
    })
}

async fn fetch_repo_files(repository: Repository) -> Result<Vec<File>> {
    let Repository {
        owner: repo_owner,
        name: repo_name,
        branch: repo_branch,
    } = repository;
    let url = format!("https://github.com/{repo_owner}/{repo_name}/archive/{repo_branch}.zip");
    let response = reqwest::get(url).await?.bytes().await?;
    let reader = std::io::Cursor::new(response);
    let mut archive = zip::ZipArchive::new(reader)?;
    let files: Vec<File> = (0..archive.len())
        .filter_map(|file| {
            let mut file = archive.by_index(file).unwrap();
            if file.is_file() {
                let mut content = String::new();
                let length = content.len();
                //Fails for non UTF-8 files
                match file.read_to_string(&mut content) {
                    Ok(_) => Some(File {
                        path: file.name().split_once("/").unwrap().1.to_string(),
                        content: content,
                        length,
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

pub async fn fetch_file_content(repository: Repository, path: &str) -> Result<String> {
    let Repository {
        owner: repo_owner,
        name: repo_name,
        branch: repo_branch,
    } = repository;
    let url =
        format!("https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{repo_branch}/{path}");
    let response = reqwest::get(url).await?;
    if response.status() == reqwest::StatusCode::OK {
        let content = response.text().await?;
        Ok(content)
    } else {
        Err(anyhow::anyhow!("Unable to fetch file content"))
    }
}

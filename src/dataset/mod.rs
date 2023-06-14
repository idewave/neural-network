use std::fs::{File, ReadDir};
use std::io::{Read, Write};

use crate::transform::image::image_to_vector;

pub fn generate_image_dataset(mut paths: Vec<&str>) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut inputs = vec![];
    let mut outputs = vec![];

    for label in 0..paths.len() {
        println!("Parsing files in: {:?}", paths[label]);
        let mut dir = std::fs::read_dir(paths[label]).unwrap();
        let input = dir.by_ref().into_iter()
            .map(|path| {
                let path = path.unwrap().path().display().to_string();
                let result: Vec<f64> = image_to_vector(&path);
                println!("GOT RESULT FOR {:?}", path);
                result
            })
            .collect::<Vec<Vec<f64>>>();

        let output = (0..input.len())
            .into_iter()
            .map(|_| label as f64)
            .collect::<Vec<f64>>();

        inputs.push(input.clone());
        outputs.push(output.clone());
    }

    (inputs.into_iter().flatten().collect(), outputs.into_iter().flatten().collect())
}

pub fn write_vector_to_file<N: serde::ser::Serialize>(
    vector: N,
    file_path: &str
) -> Result<(), std::io::Error> {
    let encoded = bincode::serialize(&vector)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let mut file = File::create(file_path)?;
    file.write_all(&encoded)?;

    Ok(())
}

pub fn read_vector_from_file<N: for<'a> serde::de::Deserialize<'a>>(
    file_path: &str
) -> Result<N, std::io::Error> {
    let mut file = File::open(file_path)?;
    let mut encoded = Vec::new();
    file.read_to_end(&mut encoded)?;

    let vector = bincode::deserialize(&encoded)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    Ok(vector)
}

// let (input, output) = generate_image_dataset(vec![
//     "./src/dataset/train/cats",
//     "./src/dataset/train/dogs"
// ]);
//
// write_vector_to_file(input, "input.bin").unwrap();
// write_vector_to_file(output, "output.bin").unwrap();
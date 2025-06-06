use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::{self, BufRead};
use std::str::FromStr;

use crate::structs::Graph;

pub fn read_instance(filename: &str) -> Result<Graph> {
    let file = File::open(filename)?;

    let reader = io::BufReader::new(file);
    let mut graph = Graph::default();

    for (row, line) in reader.lines().enumerate() {
        let sentence = line?;

        match row {
            0 => {
                let nodes = usize::from_str(extract_word(&sentence, 2)?.as_str())?;
                let edges = usize::from_str(extract_word(&sentence, 3)?.as_str())?;

                graph.matrix = vec![vec![false; nodes]; nodes];
                graph.nodes = nodes;
                graph.edges = edges;
            }
            _ => {
                let x = usize::from_str(extract_word(&sentence, 0)?.as_str())? - 1;
                let y = usize::from_str(extract_word(&sentence, 1)?.as_str())? - 1;
                graph.matrix[x][y] = true;
                graph.matrix[y][x] = true;
            }
        };
    }

    Ok(graph)
}

pub fn read_first_row(filename: &str) -> Result<i32> {
    let file = File::open(filename)?;

    let reader = io::BufReader::new(file);

    for line in reader.lines() {
        let sentence = line?;
        let nodes = i32::from_str(extract_word(&sentence, 0)?.as_str())?;
        return Ok(nodes);
    }
    Ok(0)
}

fn extract_word(sentence: &str, index: usize) -> Result<String> {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let Some(word) = words.get(index) else {
        return Err(anyhow!("Missing index on sentence: {sentence}"));
    };
    return Ok(word.to_string());
}

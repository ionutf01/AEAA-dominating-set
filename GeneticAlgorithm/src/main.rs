mod genetic_algorithm;
mod parser;
mod structs;

use anyhow::Result;
use genetic_algorithm::GeneticAlgorithm;

fn main() -> Result<()> {
    let file_name = "graph_20.gr";

    let graph = parser::read_instance(&file_name)?;
    println!("{:?}", graph);
    let mut algorithm = GeneticAlgorithm::new(graph, 100);
    algorithm.run(30)?;
    Ok(())
}

mod genetic_algorithm;
mod parser;
mod structs;

use anyhow::Result;

fn main()->Result<()> {
    let file_name = "graph_20.gr";

    let graph = parser::read_instance(&file_name)?;
    println!("{:?}", graph);
    Ok(())
}

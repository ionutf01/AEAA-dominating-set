mod genetic_algorithm;
mod parser;
mod structs;

use std::{time, u32};

use anyhow::Result;
use genetic_algorithm::GeneticAlgorithm;

fn compute_rel_err(result_size: i32, optimum_size: i32) -> f64 {
    (result_size as f64 - optimum_size as f64) / optimum_size as f64 * 100.0
}

fn main() -> Result<()> {
    let file_name = "bremen_subgraph_20.gr";
    let file_name_sol = "bremen_subgraph_20.sol";

    let graph = parser::read_instance(&file_name)?;
    let optimum_size = parser::read_first_row(&file_name_sol)?;
    println!("optimum_size: {optimum_size}");

    let mut algorithm = GeneticAlgorithm::new(graph, 100);
    let mut best_conflicts = u32::MAX;
    for i in 0..5 {
        let start = time::Instant::now();

        let (obj, conflicts, ind) = algorithm.run(100)?;
        if obj >= 1.0 {
            if conflicts < best_conflicts {
                best_conflicts = conflicts;
            }
        }
        let total_nodes = ind.solution.iter().filter(|e| **e == true).count() as i32;
        let rel_err = compute_rel_err(total_nodes, optimum_size);
        print!("{file_name}\t{}\t{}\t[", i + 1, total_nodes);
        let mut count = 0;
        for (index, node) in ind.solution.iter().enumerate() {
            if *node == true {
                print!("{}", index + 1);
                count += 1;
                if count < total_nodes {
                    print!(", ");
                }
            }
        }

        let end = time::Instant::now();
        let duration = end.duration_since(start).as_secs_f64();
        println!("]\t{:.2}\t{rel_err:.2}", duration);
    }

    Ok(())
}

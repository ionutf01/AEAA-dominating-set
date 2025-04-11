use anyhow::{anyhow, Result};
use std::{
    collections::HashSet,
    fs::File,
    io::{self, BufRead},
    mem,
};

use crate::structs::{Graph, Individual};
use rand::prelude::*;

pub struct GeneticAlgorithm {
    graph: Graph,
    population: Vec<Individual>,
    mutation_prob: f32,
}
impl Default for GeneticAlgorithm {
    fn default() -> Self {
        Self {
            graph: Default::default(),
            population: Default::default(),
            mutation_prob: Default::default(),
        }
    }
}
impl GeneticAlgorithm {
    pub fn new(graph: Graph, population_size: usize) -> Self {
        let population = vec![Individual::default(); population_size];
        let mutation_prob = 1f32 / population_size as f32;

        Self {
            graph,
            population,
            mutation_prob,
        }
    }

    fn initialize_population(&mut self) {
        self.random_initialization();
    }

    fn random_initialization(&mut self) {
        let mut rng = thread_rng();
        self.population.iter_mut().for_each(|ind| {
            let solution: Vec<bool> = (0..self.graph.nodes).map(|_| rng.gen::<bool>()).collect();
            ind.solution = solution;
        });
    }

    fn mutation(&self, ind: &Individual) -> Result<Individual> {
        // self.insert_mutation(ind)
        self.mutate(ind)
    }
    fn mutate(&self, ind: &Individual) -> Result<Individual> {
        let mut rng = rand::thread_rng();
        let mutated_solution: Vec<bool> = ind
            .solution
            .iter()
            .map(|&bit| {
                if rng.gen::<f32>() < self.mutation_prob {
                    !bit // Flip the bit
                } else {
                    bit
                }
            })
            .collect();

        Ok(Individual {
            solution: mutated_solution,
        })
    }

    fn crossover(
        &self,
        parent1: &Individual,
        parent2: &Individual,
    ) -> Result<(Individual, Individual)> {
        self.crossover_one_point(parent1, parent2)
    }
    fn crossover_one_point(
        &self,
        parent1: &Individual,
        parent2: &Individual,
    ) -> Result<(Individual, Individual)> {
        let len = parent1.solution.len();

        let mut rng = rand::thread_rng();
        let crossover_point = rng.gen_range(1..len); // Select a crossover point

        let (offspring1_solution, offspring2_solution): (Vec<bool>, Vec<bool>) = parent1
            .solution
            .iter()
            .zip(&parent2.solution)
            .enumerate()
            .map(|(i, (bit1, bit2))| {
                if i < crossover_point {
                    (*bit1, *bit2)
                } else {
                    (*bit2, *bit1)
                }
            })
            .unzip();

        Ok((
            Individual {
                solution: offspring1_solution,
            },
            Individual {
                solution: offspring2_solution,
            },
        ))
    }
    fn liniar_ranking_selection(
        &self,
        visited: &mut Vec<bool>,
        next_pop: &mut Vec<usize>,
        eval: &[f32],
    ){
        todo!();
    }

    fn evaluate(&self,ind: &Individual)->f32{
        todo!()
    }
    pub fn run(&mut self, max_iterations: usize) -> Result<()> {
        self.initialize_population();
        let mut mean = 0f32;
        let n = self.graph.nodes;
        let mut best = f32::NEG_INFINITY;
        let mut best_individual = Individual::default();

        let mut sol_min = f32::MAX;
        let mut sol_max = f32::MIN;
        let elitism=5;
        let selected=15;

        let pop_size = self.population.len();
        let mut visited = vec![false; pop_size];
        let mut eval = vec![0f32; pop_size];
        let mut next_pop = vec![0; selected];



        for iteration in 1..max_iterations {
            let mut indices: Vec<usize> = (0..pop_size).collect();
            for i in 0..pop_size {
                visited[i] = false;
                eval[i] = self.evaluate(&self.population[i]);

                if eval[i] < sol_min {
                    sol_min = eval[i];
                }
                if eval[i] > sol_max {
                    sol_max = eval[i];
                }
            }

            indices.sort_by(|&a, &b| eval[a].partial_cmp(&eval[b]).unwrap());

            if iteration % 100 == 0 {
                for &i in &indices {
                    print!("{} {:.2} ", i, eval[i]);
                }
                println!();
            }

            for i in 0..elitism {
                next_pop[i] = indices[i];
                visited[indices[i]] = true;
            }

            self.liniar_ranking_selection(&mut visited, &mut next_pop, &eval);


            let mut rng = rand::thread_rng();
            for i in 0..pop_size {
                if !visited[i] {
                    visited[i] = true;
                    for j in (i + 1)..pop_size {
                        if !visited[j] {
                            visited[j] = true;

                            // Select two random parents
                            let p1 = rng.gen_range(0..selected);
                            let mut p2 = rng.gen_range(0..selected);
                            while p1 == p2 {
                                p2 = rng.gen_range(0..selected);
                            }
                            let (c1,c2)= self.crossover(&self.population[next_pop[p1]],&self.population[next_pop[p2]])?;
                            self.population[i] = c1;
                            self.population[j] = c2;

                            self.population[i] = self.mutate(&self.population[i])?;
                            self.population[j] = self.mutate(&self.population[j])?;

                            break;
                        }
                    }
                }
            }
        }
        // for iter in (0..max_iter){

        //     let obj = self.evaluate_objective(&individual.tour, &individual.packing_plan)?;
        //     println!("[Obj]{}", obj);
        //     mean += obj;
        //     if obj > best {
        //         best = obj;
        //         best_individual = individual.clone();
        //     }
        // }
        // mean /= n as f32;
        // println!("[Mean:]{:?}, [Best]{:?}", mean, best);

        Ok(())
    }
}

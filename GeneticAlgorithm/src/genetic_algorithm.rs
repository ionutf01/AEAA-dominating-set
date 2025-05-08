use anyhow::Result;

use crate::structs::{Graph, Individual};
use rand::prelude::*;

pub struct GeneticAlgorithm {
    graph: Graph,
    population: Vec<Individual>,
    mutation_prob: f64,
    n_ds: usize,
    n_core: usize,
}
impl Default for GeneticAlgorithm {
    fn default() -> Self {
        Self {
            graph: Default::default(),
            population: Default::default(),
            mutation_prob: Default::default(),
            n_ds: Default::default(),
            n_core: Default::default(),
        }
    }
}
impl GeneticAlgorithm {
    pub fn new(graph: Graph, population_size: usize) -> Self {
        let population = vec![Individual::default(); population_size];
        // let mutation_prob = 1f64 / population_size as f64;
        let mutation_prob = 0.01;
        let n_ds = 10;
        let n_core = 3;

        Self {
            graph,
            population,
            mutation_prob,
            n_ds,
            n_core,
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
                if rng.gen::<f64>() < self.mutation_prob {
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

    fn evaluate(&self, ind: &Individual) -> f64 {
        let total_nodes = self.graph.nodes;
        let mut covered_nodes = 0;
        let mut contained_nodes = 0;
        let mut covered_list = vec![false; total_nodes];
        for (i, &node_in_solution) in ind.solution.iter().enumerate() {
            if node_in_solution == true {
                contained_nodes += 1;
                covered_nodes += 1;
                covered_list[i] = true;

                // adjacent nodes that are covered
                for j in 0..total_nodes {
                    if self.graph.matrix[i][j] == true
                        && ind.solution[j] == false
                        && covered_list[j] == false
                    {
                        covered_nodes += 1;
                        covered_list[j] = true;
                    }
                }
            }
        }
        // println!("[covered]{:?}, {:?}", covered_nodes, total_nodes);
        let fitness = (covered_nodes as f64 / total_nodes as f64)
            + (1.0 / (total_nodes as f64 * contained_nodes as f64));

        fitness
    }

    fn local_search(&self, ind: &Individual, max_iterations: usize) -> Individual {
        let mut rng = rand::thread_rng();
        let total_nodes = self.graph.nodes;
        let mut ind = ind.clone();
        let mut best_fitness = self.evaluate(&ind);

        for _ in 0..max_iterations {
            let mut x_best_temp = ind.clone();
            let fitness = self.evaluate(&x_best_temp);

            if fitness >= 1.0 {
                // Randomly select a component with value 1, inversely proportional to its degree
                let mut degrees = vec![0; total_nodes];
                for i in 0..total_nodes {
                    degrees[i] = self.graph.matrix[i].iter().map(|b| *b as i32).sum();
                }

                let mut weighted_indices: Vec<(usize, f64)> = x_best_temp
                    .solution
                    .iter()
                    .enumerate()
                    .filter(|&(_, &value)| value == true)
                    .map(|(index, _)| (index, 1.0 / degrees[index] as f64))
                    .collect();

                let total_weight: f64 = weighted_indices.iter().map(|&(_, weight)| weight).sum();
                let r = rng.gen_range(0.0..total_weight);
                let mut u = 0.0;

                for (index, weight) in weighted_indices.iter_mut() {
                    u += *weight;
                    if r < u {
                        x_best_temp.solution[*index] = false;
                        break;
                    }
                }
            } else {
                // Randomly select a component with value 0, proportional to its degree
                let mut degrees = vec![0; total_nodes];
                for i in 0..total_nodes {
                    degrees[i] = self.graph.matrix[i].iter().map(|b| *b as i32).sum();
                }

                let mut weighted_indices: Vec<(usize, f64)> = x_best_temp
                    .solution
                    .iter()
                    .enumerate()
                    .filter(|&(_, &value)| value == false)
                    .map(|(index, _)| (index, degrees[index] as f64))
                    .collect();

                let total_weight: f64 = weighted_indices.iter().map(|&(_, weight)| weight).sum();
                let r = rng.gen_range(0.0..total_weight);
                let mut u = 0.0;

                for (index, weight) in weighted_indices.iter_mut() {
                    u += *weight;
                    if r < u {
                        x_best_temp.solution[*index] = true;
                        break;
                    }
                }
            }

            if fitness >= best_fitness {
                ind = x_best_temp;
                best_fitness = fitness;
            }
        }
        ind
    }

    fn update_dominating_sets(
        &self,
        dominating_sets: &mut Vec<(f64, Individual)>,
        new_ind: &Individual,
    ) {
        let new_obj = self.evaluate(new_ind);
        let mut index = None;
        // find if the new solution is better than any existing solutions.
        // dominating sets is in inverse order of the objective function
        for (i, (obj, _)) in dominating_sets.iter().rev().enumerate() {
            if new_obj < *obj {
                index = Some(i);
            }
        }

        match index {
            Some(i) => {
                // don't update if the solution is not better than any already existing ones
                if i != self.n_ds - 1 {
                    dominating_sets.insert(i + 1, (new_obj, new_ind.clone()));
                }
            }
            // the new solution is best so far and is inserted on the first position
            None => {
                dominating_sets.insert(0, (new_obj, new_ind.clone()));
            }
        }
        // truncate to the predetermined size if needed
        if dominating_sets.len() > self.n_ds {
            dominating_sets.resize(self.n_ds, (new_obj, new_ind.clone()));
        }
    }

    fn filtering(&self, ind: &Individual) -> Option<Individual> {
        let mut fitness = self.evaluate(&ind);

        if fitness < 1.0 {
            return None;
        }
        let mut best_ind: Individual = ind.clone();
        let mut intersection: Vec<usize> = Vec::new();
        for (i, &value) in best_ind.solution.iter().enumerate() {
            if value == true {
                intersection.push(i);
            }
        }

        // check if any node could be removed without damaging the cover
        for &j in &intersection {
            let original_value = best_ind.solution[j];
            best_ind.solution[j] = false;

            let new_fitness = self.evaluate(&best_ind);

            if new_fitness >= fitness {
                fitness = new_fitness;
                continue;
            } else {
                best_ind.solution[j] = original_value;
            }
        }
        Some(best_ind)
    }

    fn elite_inspiration(
        &self,
        dominating_sets: &mut Vec<(f64, Individual)>,
        best_ind: &Individual,
    ) -> Option<Individual> {
        if dominating_sets.is_empty() {
            return None;
        }
        let n_f = best_ind.solution.iter().filter(|v| **v == true).count();
        let mut intersection = vec![true; best_ind.solution.len()];

        dominating_sets
            .iter()
            .take(self.n_core)
            .for_each(|(_, ind)| {
                ind.solution.iter().enumerate().for_each(|(index, value)| {
                    intersection[index] = intersection[index] & (*value);
                });
            });
        let mut x_core = Individual {
            solution: intersection,
        };
        println!("[x_core]{:?}", x_core);

        loop {
            let n_core = x_core.solution.iter().filter(|v| **v == true).count();
            if n_core >= n_f - 1 {
                return None;
            }
            if self.evaluate(&x_core) >= 1.0 {
                return Some(x_core);
            }
            let mut best_improvement = None;
            let mut best_pos = None;
            for (i, val) in x_core.solution.iter().enumerate() {
                if *val == false {
                    let mut temp = x_core.clone();
                    temp.solution[i] = true;
                    let obj = self.evaluate(&temp);
                    match best_improvement {
                        Some(best_obj) => {
                            if obj > best_obj {
                                best_improvement = Some(obj);
                                best_pos = Some(i);
                            }
                        }
                        None => {}
                    }
                }
            }
            match best_pos {
                Some(pos) => x_core.solution[pos] = true,
                None => {
                    break;
                }
            }
        }
        Some(x_core)
    }

    pub fn run(&mut self, max_iterations: usize) -> Result<()> {
        // let nodes = vec![2, 5, 9, 11, 14, 15, 17, 19, 22, 23, 25, 26, 28, 31, 32];
        // let mut ind = Individual::new(self.graph.nodes);
        // for i in nodes {
        //     ind.solution[i - 1] = true;
        // }
        // let obj = self.evaluate(&ind);
        // println!("[ind]{:?}", ind);

        // println!("[Obj]{}", obj);
        // if let Some(best) = self.filtering(&ind) {
        //     let obj = self.evaluate(&best);
        //     println!("[ind]{:?}", ind);

        //     println!("[Obj]{}", obj);
        // }

        // panic!();
        self.initialize_population();
        let mut mean = 0f64;
        let mut best = f64::NEG_INFINITY;
        let mut best_individual = Individual::default();

        let elitism = 5;
        let selected = 15;

        let pop_size = self.population.len();
        let mut visited = vec![false; pop_size];
        let mut eval = vec![0f64; pop_size];
        let mut next_pop = vec![0; selected];

        let mut dominating_sets: Vec<(f64, Individual)> = Vec::new();

        for iteration in 0..max_iterations {
            let mut indices: Vec<usize> = (0..pop_size).collect();
            for i in 0..pop_size {
                visited[i] = false;
                eval[i] = self.evaluate(&self.population[i]);
            }

            indices.sort_by(|&a, &b| eval[b].partial_cmp(&eval[a]).unwrap());

            if iteration % 100 == 0 {
                for &i in &indices {
                    print!("{} {:.2} ", i, eval[i]);
                }
                println!();
            }

            // keep the best individuals
            for i in 0..elitism {
                next_pop[i] = indices[i];
                visited[indices[i]] = true;
            }
            let obj = self.evaluate(&self.population[next_pop[0]]);

            mean += obj;
            if obj > best {
                best = obj;
                best_individual = self.population[next_pop[0]].clone();
            }

            // select individuals for the next generation
            let ranks = linear_ranking_selection(self.population.len(), 10);

            for (i, rank) in ranks.into_iter().enumerate() {
                next_pop[elitism + i] = rank;
                visited[rank] = true;
            }

            let mut rng = rand::thread_rng();
            // apply operators over the pool of chosen individuals
            for i in 0..pop_size {
                if !visited[i] {
                    // replace individuals that were not selected
                    visited[i] = true;
                    for j in (i + 1)..pop_size {
                        if !visited[j] {
                            // find the next availabe spot for replacement
                            visited[j] = true;

                            // select two random parents from the pool
                            let p1 = rng.gen_range(0..selected);
                            let mut p2 = rng.gen_range(0..selected);
                            while p1 == p2 {
                                p2 = rng.gen_range(0..selected);
                            }
                            // apply operators
                            let (c1, c2) = self.crossover(
                                &self.population[next_pop[p1]],
                                &self.population[next_pop[p2]],
                            )?;
                            self.population[i] = c1;
                            self.population[j] = c2;

                            self.population[i] = self.mutate(&self.population[i])?;
                            self.population[j] = self.mutate(&self.population[j])?;

                            self.population[i] = self.local_search(&self.population[i], 20);
                            self.population[j] = self.local_search(&self.population[j], 20);

                            let obj_i = self.evaluate(&self.population[i]);
                            let obj_j = self.evaluate(&self.population[j]);
                            // store the best dominating sets for the final step
                            self.update_dominating_sets(&mut dominating_sets, &self.population[i]);
                            self.update_dominating_sets(&mut dominating_sets, &self.population[j]);

                            // applyy filtering
                            if let Some(ind) = self.filtering(&self.population[i]) {
                                self.population[i] = ind;
                                // update the dominating sets if needed
                                if self.evaluate(&self.population[i]) > obj_i {
                                    self.update_dominating_sets(
                                        &mut dominating_sets,
                                        &self.population[i],
                                    );
                                }
                            }
                            if let Some(ind) = self.filtering(&self.population[j]) {
                                self.population[j] = ind;
                                // update the dominating sets if needed
                                if self.evaluate(&self.population[j]) > obj_j {
                                    self.update_dominating_sets(
                                        &mut dominating_sets,
                                        &self.population[j],
                                    );
                                }
                            }

                            break;
                        }
                    }
                }
            }
        }

        let obj = self.evaluate(&best_individual);
        println!("[Obj]{}", obj);
        let conflicts = self.count_conflicts(&best_individual);
        println!("[conflicts]{}", conflicts);

        for (index, node) in best_individual.solution.iter().enumerate() {
            if *node == true {
                print!("{}, ", index + 1);
            }
        }

        if let Some(best) = self.filtering(&best_individual) {
            best_individual = best;
            println!("[]{:?}", best_individual);
            let obj = self.evaluate(&best_individual);
            println!("[Obj]{}", obj);
            let conflicts = self.count_conflicts(&best_individual);
            println!("[conflicts]{}", conflicts);
            for (index, node) in best_individual.solution.iter().enumerate() {
                if *node == true {
                    print!("{}, ", index + 1);
                }
            }
        }
        // println!("[dominating_sets]{:?}", dominating_sets);
        // println!("[dominating_sets]{:?}", dominating_sets.len());
        // let elite = self.elite_inspiration(&mut dominating_sets, &best_individual);
        // println!("[elite]{:?}", elite);

        // if let Some(elite) = elite {
        //     let obj_elite = self.evaluate(&elite);
        //     println!("[obj_elite]{:?}", obj_elite);
        // }

        Ok(())
    }
    pub fn count_conflicts(&self, ind: &Individual) -> u32 {
        let mut count = 0u32;
        let total_nodes = self.graph.nodes;
        for (i, &value) in ind.solution.iter().enumerate() {
            if value == true {
                // count neighbors that are also in the set
                for j in 0..total_nodes {
                    if self.graph.matrix[i][j] == true && ind.solution[j] == true {
                        count += 1;
                        println!("[conflict]{i}, {j}");
                    }
                }
            }
        }
        count
    }
}
fn linear_ranking_selection(population_size: usize, selection_number: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut selected_indices = Vec::new();

    let total_rank_sum: f64 = (1..=population_size).sum::<usize>() as f64;

    for _ in 0..selection_number {
        let r = rng.gen_range(0.0..total_rank_sum);
        let mut u = 0.0;

        for i in 0..population_size {
            u += (i + 1) as f64; // Rank-based probability
            if r < u {
                selected_indices.push(i);
                break;
            }
        }
    }

    selected_indices
}

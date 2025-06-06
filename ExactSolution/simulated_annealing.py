import random
import math
import os
import time
import argparse

def is_dominating_set(adj_list, candidate_set):
    dominated = set(candidate_set)
    for node in candidate_set:
        dominated.update(adj_list[node])
    return len(dominated) == len(adj_list)

def domination_penalty_cost(candidate_set, adj_list, penalty_weight=1000):
    dominated = set(candidate_set)
    for node in candidate_set:
        dominated.update(adj_list[node])
    uncovered = len(adj_list) - len(dominated)
    return len(candidate_set) + penalty_weight * uncovered

def get_neighbors(solution, adj_list, coverage_count):
    neighbor = solution.copy()
    vertices = set(adj_list.keys())

    if random.random() < 0.5:
        redundant = [v for v in neighbor if all(coverage_count[u] > 1 for u in adj_list[v] | {v})]
        if redundant:
            remove = random.choice(redundant)
            neighbor.remove(remove)
    else:
        dominated = set(neighbor)
        for node in neighbor:
            dominated.update(adj_list[node])
        candidates = list(vertices - dominated)
        if candidates:
            best = max(candidates, key=lambda x: len(adj_list[x] - dominated))
            neighbor.add(best)
    return neighbor

def greedy_initial_solution(adj_list):
    uncovered = set(adj_list.keys())
    solution = set()
    while uncovered:
        best = max(adj_list.keys(), key=lambda x: len(adj_list[x] & uncovered))
        solution.add(best)
        uncovered -= adj_list[best] | {best}
    return solution

def compute_coverage_count(adj_list, dominating_set):
    coverage_count = {v: 0 for v in adj_list}
    for v in dominating_set:
        coverage_count[v] += 1
        for u in adj_list[v]:
            coverage_count[u] += 1
    return coverage_count

def simulated_annealing(adj_list, initial_temp=500, cooling_rate=0.995, max_iter=20000, restarts=5):
    best_overall = None

    for _ in range(restarts):
        current_solution = greedy_initial_solution(adj_list)
        best_solution = current_solution.copy()
        temp = initial_temp

        for iteration in range(max_iter):
            coverage_count = compute_coverage_count(adj_list, current_solution)
            neighbor = get_neighbors(current_solution, adj_list, coverage_count)

            cost_current = domination_penalty_cost(current_solution, adj_list)
            cost_neighbor = domination_penalty_cost(neighbor, adj_list)

            if cost_neighbor < cost_current or random.random() < math.exp((cost_current - cost_neighbor) / temp):
                current_solution = neighbor
                if domination_penalty_cost(current_solution, adj_list) < domination_penalty_cost(best_solution, adj_list):
                    best_solution = current_solution

            temp *= cooling_rate

        if best_overall is None or domination_penalty_cost(best_solution, adj_list) < domination_penalty_cost(best_overall, adj_list):
            best_overall = best_solution

    return best_overall

def parse_gr_file(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    adj_list = {}
    for line in lines:
        if line.startswith('p'):
            _, _, n, _ = line.strip().split()
            for i in range(1, int(n) + 1):
                adj_list[i] = set()
        elif not line.startswith('c'):
            u, v = map(int, line.strip().split())
            adj_list[u].add(v)
            adj_list[v].add(u)
    return adj_list

def read_solution_size(sol_filepath):
    try:
        with open(sol_filepath) as f:
            lines = f.readlines()
        for line in lines:
            if line.strip().isdigit():
                return int(line.strip())
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Simulated Annealing for Dominating Set problem")
    parser.add_argument("graph_file", help="Name of the .gr file inside the TestInstances folder")
    parser.add_argument("--initial_temp", type=float, default=500.0, help="Initial temperature for annealing")
    parser.add_argument("--cooling_rate", type=float, default=0.995, help="Cooling rate for annealing")
    args = parser.parse_args()

    graph_path = os.path.join("TestInstances", args.graph_file)
    sol_path = graph_path.replace(".gr", ".sol")

    graph = parse_gr_file(graph_path)
    start = time.time()
    result = simulated_annealing(graph, initial_temp=args.initial_temp, cooling_rate=args.cooling_rate)
    end = time.time()

    result_size = len(result)
    optimum_size = read_solution_size(sol_path)

    print(f"Dominating set size: {result_size}")
    print(f"Dominating set: {sorted(result)}")
    print(f"Took {end - start:.2f}s")

    if optimum_size is not None:
        rel_error = (result_size - optimum_size) / optimum_size * 100
        print(f"Relative error: {rel_error:.2f}%")
    else:
        print("Optimum solution not found for comparison.")

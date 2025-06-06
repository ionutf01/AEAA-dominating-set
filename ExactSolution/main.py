import sys
import time
import os
from itertools import combinations

def k_dominating_set(graph_file, k, i, j):
    """
    Determines if a graph has a dominating set of size at most k,
    reading the graph from a DIMACS-like .gr format file.

    Args:
        graph_file: Path to the .gr format graph file.
        k: The size of the dominating set to find.
        i: Parameter i from K_{i,j}-free graph definition.
        j: Parameter j from K_{i,j}-free graph definition.

    Returns:
        A list representing the dominating set, or None if no such
        dominating set exists. The first element in the list is the size
        of the dominating set.
    """
    print(">>> ENTERED k_dominating_set <<<")
    graph = read_graph_from_file(graph_file)
    if not graph:
        return None

    t_start = time.time()
    rwb_graph = initialize_rwb_graph(graph)
    print(f"[Time] initialize_rwb_graph: {time.time() - t_start:.3f} sec")

    t_start = time.time()
    rwb_graph = apply_rule_1(rwb_graph)
    print(f"[Time] apply_rule_1: {time.time() - t_start:.3f} sec")
    print(f"[After Rule 1] Vertex count: {len(rwb_graph)}")
    print(f"[Stats] Colors: {sum(1 for d in rwb_graph.values() if d['color'] == 'red')} red, "
      f"{sum(1 for d in rwb_graph.values() if d['color'] == 'white')} white, "
      f"{sum(1 for d in rwb_graph.values() if d['color'] == 'blue')} blue")

    for p in range(1, i - 1):
        while True:
            t_start = time.time()
            modified_graph = rule_2_p(rwb_graph, k, i, j, p)
            duration = time.time() - t_start
            if modified_graph == rwb_graph:
                break
            rwb_graph = modified_graph
    print(f"[Time] rule_2_p (p={p}): {duration:.3f} sec")
    print(f"[After Rule 2_p] Vertex count: {len(rwb_graph)}")
    print(f"[Stats] Colors: {sum(1 for d in rwb_graph.values() if d['color'] == 'red')} red, "
        f"{sum(1 for d in rwb_graph.values() if d['color'] == 'white')} white, "
        f"{sum(1 for d in rwb_graph.values() if d['color'] == 'blue')} blue")

    while True:
        t_start = time.time()
        modified_graph = apply_rule_3(rwb_graph, k, i, j)
        duration = time.time() - t_start
        if modified_graph == rwb_graph:
            break
        rwb_graph = modified_graph

    print(f"[Time] apply_rule_3: {duration:.3f} sec")
    print(f"[After Rule 3] Vertex count: {len(rwb_graph)}")
    print(f"[Stats] Colors: {sum(1 for d in rwb_graph.values() if d['color'] == 'red')} red, "
        f"{sum(1 for d in rwb_graph.values() if d['color'] == 'white')} white, "
        f"{sum(1 for d in rwb_graph.values() if d['color'] == 'blue')} blue")

    while True:
        t_start = time.time()
        modified_graph = apply_rule_4(rwb_graph)
        duration = time.time() - t_start
        if modified_graph == rwb_graph:
            break
        rwb_graph = modified_graph

    print(f"[Time] apply_rule_4: {duration:.3f} sec")
    print(f"[After Rule 4] Vertex count: {len(rwb_graph)}")
    print(f"[Stats] Colors: {sum(1 for d in rwb_graph.values() if d['color'] == 'red')} red, "
        f"{sum(1 for d in rwb_graph.values() if d['color'] == 'white')} white, "
        f"{sum(1 for d in rwb_graph.values() if d['color'] == 'blue')} blue")

    while True:
        t_start = time.time()
        modified_graph = apply_rule_5(rwb_graph)
        duration = time.time() - t_start
        if modified_graph == rwb_graph:
            break
        rwb_graph = modified_graph

    print(f"[Time] apply_rule_5: {duration:.3f} sec")
    print(f"[After Rule 5] Vertex count: {len(rwb_graph)}")
    print(f"[Stats] Colors: {sum(1 for d in rwb_graph.values() if d['color'] == 'red')} red, "
        f"{sum(1 for d in rwb_graph.values() if d['color'] == 'white')} white, "
        f"{sum(1 for d in rwb_graph.values() if d['color'] == 'blue')} blue")

    t_start = time.time()
    result = apply_rule_6(rwb_graph, k, i, j)
    print(f"[Time] apply_rule_6: {time.time() - t_start:.3f} sec")
    print(f"[After Rule 6] Vertex count: {len(rwb_graph)}")
    print(f"[Stats] Colors: {sum(1 for d in rwb_graph.values() if d['color'] == 'red')} red, "
      f"{sum(1 for d in rwb_graph.values() if d['color'] == 'white')} white, "
      f"{sum(1 for d in rwb_graph.values() if d['color'] == 'blue')} blue")

    if result == "NO":
        return None

    t_start = time.time()
    dominating_set = find_dominating_set(rwb_graph, k)
    print(f"[Time] find_dominating_set: {time.time() - t_start:.3f} sec")

    return dominating_set

def read_graph_from_file(graph_file):
    """
    Reads a graph from a DIMACS-like .gr format file.

    Args:
        graph_file: Path to the .gr format graph file.

    Returns:
        A graph represented as an adjacency list, or None on error.
    """
    graph = {}
    try:
        with open(graph_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue  # Skip comments and empty lines

                parts = line.split()
                if parts[0] == 'p':  # Problem descriptor line
                    if len(parts) != 4 or parts[1] != 'ds':
                        print("Error: Invalid problem descriptor line.")
                        return None
                    num_vertices = int(parts[2])
                    # Initialize graph with empty adjacency lists
                    for v in range(1, num_vertices + 1):
                        graph[v] = []
                else:  # Edge line
                    if len(parts) != 2:
                        print("Error: Invalid edge line.")
                    u, v = int(parts[0]), int(parts[1])
                    if u not in graph or v not in graph:
                        print("Error: Vertex number out of range.")
                        return None
                    graph[u].append(v)
                    graph[v].append(u)  # Assuming undirected graph
        return graph
    except FileNotFoundError:
        print(f"Error: File not found: {graph_file}")
        return None
    except ValueError:
        print("Error: Invalid data in file.")
        return None

def initialize_rwb_graph(graph):
    """
    Initializes the graph as an rwb-graph by coloring all vertices blue.

    Args:
        graph: A graph represented as an adjacency list.

    Returns:
        An rwb-graph where all vertices are blue.
        An rwb-graph is a dictionary where:
        - Keys are vertices.
        - Values are dictionaries with 'neighbors' (list of adjacent vertices)
          and 'color' ('red', 'white', or 'blue').
    """
    rwb_graph = {}
    for vertex in graph:
        rwb_graph[vertex] = {
            'neighbors': graph[vertex],
            'color': 'blue'
        }
    return rwb_graph

def apply_rule_1(rwb_graph):
    """
    Applies Rule 1: Color all isolated blue vertices of G red.

    Args:
        rwb_graph: The input rwb-graph.

    Returns:
        The modified rwb-graph.
    """
    for vertex, data in rwb_graph.items():
        if data['color'] == 'blue' and not data['neighbors']:
            rwb_graph[vertex]['color'] = 'red'
    return rwb_graph

def rule_2_p(rwb_graph, k, i, j, p):
    """
    Applies Rule 2.p: Handles complex reduction based on common blue neighbors.

    Args:
        rwb_graph: The input rwb-graph.
        k: The parameter k.
        i: Parameter i from K_{i,j}-free graph definition.
        j: Parameter j from K_{i,j}-free graph definition.
        p: The current value of p.

    Returns:
        The modified rwb-graph.
    """
    if p == 1:
        b = j * k
    else:
        b = j * (k ** p) + sum(k ** q for q in range(1, p))

    vertices = list(rwb_graph.keys())
    for U in combinations(vertices, i - p):
        if all(rwb_graph[u]['color'] != 'red' for u in U):
            common_blue_neighbors = find_common_blue_neighbors(rwb_graph, U)
            if len(common_blue_neighbors) > b:
                # Apply the rule
                # 1. Add gadget vertices X and edges
                X = [f"x_{u_idx}_{p}" for u_idx in range(len(U))]  # Unique gadget vertex names
                for u_idx, u in enumerate(U):
                    for x in X:
                        rwb_graph[u]['neighbors'].append(x)
                        rwb_graph[x] = {'neighbors': [u], 'color': 'blue'}

                # 2. Color vertices in B white
                for blue_neighbor in common_blue_neighbors:
                    rwb_graph[blue_neighbor]['color'] = 'white'

                # 3. Color vertices in X blue
                for x in X:
                    rwb_graph[x]['color'] = 'blue'
                return rwb_graph
    return rwb_graph

def find_common_blue_neighbors(rwb_graph, U):
    """
    Finds the common blue neighbors of a set of vertices U.

    Args:
        rwb_graph: The input rwb-graph.
        U: A set of vertices.

    Returns:
        A list of common blue neighbors.
    """
    if not U:
        return []

    common_neighbors = set(rwb_graph[U[0]]['neighbors'])
    for u in U[1:]:
        neighbors = set(rwb_graph[u]['neighbors'])
        common_neighbors.intersection_update(neighbors)

    return [v for v in common_neighbors if rwb_graph[v]['color'] == 'blue']

def apply_rule_3(rwb_graph, k, i, j):
    """
    Applies Rule 3: If a blue or white vertex has too many blue neighbors, color it red.

    Args:
        rwb_graph: The input rwb-graph.
        k: The parameter k.
        i: Parameter i from K_{i,j}-free graph definition.
        j: Parameter j from K_{i,j}-free graph definition.

    Returns:
        The modified rwb-graph.
    """
    h = j * (k ** (i - 1)) + sum(k ** q for q in range(2, i))

    for vertex, data in rwb_graph.items():
        if data['color'] in ('blue', 'white'):
            blue_neighbors = [
                neighbor
                for neighbor in data['neighbors']
                if rwb_graph[neighbor]['color'] == 'blue'
            ]
            if len(blue_neighbors) > h:
                rwb_graph[vertex]['color'] = 'red'
                for neighbor in blue_neighbors:
                    rwb_graph[neighbor]['color'] = 'white'
                return rwb_graph
    return rwb_graph

def apply_rule_4(rwb_graph):
    """
    Applies Rule 4: Delete white vertices with at most one blue neighbor.

    Args:
        rwb_graph: The input rwb-graph.

    Returns:
        The modified rwb-graph.
    """
    vertices_to_delete = []
    for vertex, data in rwb_graph.items():
        if data['color'] == 'white':
            blue_neighbors = [
                neighbor
                for neighbor in data['neighbors']
                if rwb_graph[neighbor]['color'] == 'blue'
            ]
            if len(blue_neighbors) <= 1:
                vertices_to_delete.append(vertex)

    for vertex in vertices_to_delete:
        neighbors = rwb_graph[vertex]['neighbors']
        del rwb_graph[vertex]
        for neighbor in neighbors:
            if neighbor in rwb_graph:
                rwb_graph[neighbor]['neighbors'].remove(vertex)
    if vertices_to_delete:
        return apply_rule_1(rwb_graph)
    return rwb_graph

def apply_rule_5(rwb_graph):
    """
    Applies Rule 5: Delete white vertices whose blue neighborhood is contained
    in the blue neighborhood of another white/blue vertex.

    Args:
        rwb_graph: The input rwb-graph.

    Returns:
        The modified rwb-graph.
    """
    vertices_to_delete = []
    for u, u_data in rwb_graph.items():
        if u_data['color'] == 'white':
            for v, v_data in rwb_graph.items():
                if u != v and v_data['color'] in ('white', 'blue'):
                    u_blue_neighbors = set(
                        n for n in u_data['neighbors'] if rwb_graph[n]['color'] == 'blue'
                    )
                    v_blue_neighbors = set(
                        n for n in v_data['neighbors'] if rwb_graph[n]['color'] == 'blue'
                    )
                    if u_blue_neighbors.issubset(v_blue_neighbors):
                        vertices_to_delete.append(u)
                        break

    for vertex in vertices_to_delete:
        neighbors = rwb_graph[vertex]['neighbors']
        del rwb_graph[vertex]
        for neighbor in neighbors:
            if neighbor in rwb_graph:
                rwb_graph[neighbor]['neighbors'].remove(vertex)
    if vertices_to_delete:
        return apply_rule_1(rwb_graph)
    return rwb_graph

def apply_rule_6(rwb_graph, k, i, j):
    """
    Applies Rule 6: Checks the number of red and blue vertices.

    Args:
        rwb_graph: The input rwb-graph.
        k: The parameter k.
        i: Parameter i from K_{i,j}-free graph definition.
    Returns:
        "NO" if the graph does not have a dominating set, otherwise the
        modified rwb_graph.
    """
    red_count = sum(1 for data in rwb_graph.values() if data['color'] == 'red')
    blue_count = sum(1 for data in rwb_graph.values() if data['color'] == 'blue')

    if red_count > k or blue_count > j * (k ** i) + sum(k ** q for q in range(2, i + 1)):
        return "NO"
    return rwb_graph

def all_subsets(vertices, size):
    """
    Generates all subsets of 'vertices' of a given 'size'.

    Args:
        vertices: A list of vertices.
        size: The size of the subsets to generate.

    Returns:
        A list of all subsets.
    """
    if size == 0:
        return [[]]
    if not vertices:
        return []

    subsets = []
    for i in range(len(vertices)):
        first = [vertices[i]]
        rest = vertices[i + 1:]
        for subset in all_subsets(rest, size - 1):
            subsets.append(first + subset)
    return subsets

def find_dominating_set_test(rwb_graph, k):
    """
    Tries to find the dominating set {3, 6} if it exists.
    """
    vertices = list(rwb_graph.keys())

    def is_dominating_set(subset):
        dominated = set()
        for vertex in subset:
            dominated.add(vertex)
            dominated.update(rwb_graph[vertex]['neighbors'])
        return len(dominated) == len(rwb_graph)

    # Check if {3, 6} is a dominating set
    subset = [3, 6]
    if all(v in vertices for v in subset) and len(subset) <= k and is_dominating_set(subset):
        return [len(subset)] + subset

    # Fallback to the original backtracking
    def backtrack(index, current_subset):
        if len(current_subset) > k:
            return None
        if index == len(vertices):
            if is_dominating_set(current_subset):
                return [len(current_subset)] + current_subset
            else:
                return None

        result = backtrack(index + 1, current_subset + [vertices[index]])
        if result:
            return result

        result = backtrack(index + 1, current_subset)
        if result:
            return result
        return None

    return backtrack(0, [])

def find_dominating_set(rwb_graph, k):
    """
    Finds a dominating set of size at most k in the reduced graph using backtracking.
    """
    vertices = list(rwb_graph.keys())

    def is_dominating_set(subset):
        """
        Checks if a subset of vertices is a dominating set.
        """
        dominated = set()
        for vertex in subset:
            dominated.add(vertex)
            dominated.update(rwb_graph[vertex]['neighbors'])
        return len(dominated) == len(rwb_graph)

    def backtrack(index, current_subset):
        """
        Backtracking function to find a dominating set.
        """
        if len(current_subset) > k:
            return None  # Exceeded the maximum allowed size
        if len(current_subset) + (len(vertices) - index) < k:
            return None
        if index == len(vertices):
            if is_dominating_set(current_subset):
                print(f"Found dominating set of size {len(current_subset)}, vertices: {current_subset}")
                return [len(current_subset)] + list(current_subset)  # Ensure current_subset is a list
            else:
                return None

        # Option 1: Include the current vertex in the subset
        result = backtrack(index + 1, current_subset + [vertices[index]])
        if result:
            return result

        # Option 2: Exclude the current vertex from the subset
        result = backtrack(index + 1, current_subset)
        if result:
            return result
        return None

    vertices = list(rwb_graph.keys())
    print("Order of vertices:", vertices)
    return backtrack(0, [])

def read_solution_file(sol_file):
    """
    Reads the expected dominating set from a .sol file.

    Args:
        sol_file: Path to the .sol file.

    Returns:
        A list representing the expected dominating set, or None on error.
    """
    try:
        with open(sol_file, 'r') as f:
            size = int(f.readline().strip())
            dominating_set = [int(line.strip()) for line in f]
            return [size] + dominating_set
    except FileNotFoundError:
        print(f"Error: Solution file not found: {sol_file}")
        return None
    except ValueError:
        print("Error: Invalid data in solution file.")
        return None

def is_kij_free(graph, i, j):
    """
    Checks if the graph is K_{i,j}-free.

    Args:
        graph: Dict[int, List[int]] - Adjacency list
        i: int - size of the first part
        j: int - size of the second part

    Returns:
        True if the graph is K_{i,j}-free, False otherwise
    """
    vertices = list(graph.keys())
    
    for A in combinations(vertices, i):
        # Get the common neighbors of all vertices in A
        common_neighbors = set(graph[A[0]])
        for a in A[1:]:
            common_neighbors &= set(graph[a])
        
        if len(common_neighbors) < j:
            continue  # Not enough candidates for B

        for B in combinations(common_neighbors, j):
            # Check if all edges exist between each a in A and each b in B
            if all(b in graph[a] for a in A for b in B):
                print(f"Found K_{{{i},{j}}} between A={A} and B={B}")
                return False  # Not K_{i,j}-free

    return True  # No such bipartite complete subgraph found

if __name__ == "__main__":
    graph_file = "bremen_subgraph_50.gr"
    sol_file = "bremen_subgraph_50.sol"

    graph_path = os.path.join("TestInstances", graph_file)
    sol_path = os.path.join("TestInstances", sol_file)

    k = 9
    i = 3
    j = 2

    start_time = time.time()
    result = k_dominating_set(graph_path, k, i, j)
    end_time = time.time()

    print(">>> FINISHED CALL TO k_dominating_set <<<")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")


    expected_solution = read_solution_file(sol_path)

    if result:
        size = result[0]
        dominating_set_vertices = result[1:]

        print("Dominating Set Size:", size)
        print("Dominating Set Vertices:", dominating_set_vertices)

        if expected_solution:
            expected_size = expected_solution[0]
            expected_vertices = expected_solution[1:]

            print("Expected Dominating Set Size:", expected_size)
            print("Expected Dominating Set Vertices:", expected_vertices)

            if dominating_set_vertices == expected_vertices:
                print("The found dominating set matches the expected solution.")
            else:
                print("The found dominating set does NOT match the expected solution.")
        else:
            print("Could not read the expected solution for comparison.")
    else:
        print("No dominating set of the specified size exists.")
        if expected_solution:
            expected_size = expected_solution[0]
            expected_vertices = expected_solution[1:]
            print("Expected Dominating Set Size:", expected_size)
            print("Expected Dominating Set Vertices:", expected_vertices)
        else:
            print("Could not read the expected solution for comparison.")

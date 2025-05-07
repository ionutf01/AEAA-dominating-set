import sys

# Global cache for memoization
# Keys: (canonical_representation_of_current_S_tuples, frozenset_of_U_target_to_cover)
# Values: (cost, set_of_chosen_original_vertex_ids_for_dominating_set)
memo = {}

def parse_gr_file(filepath):
    """
    Parses a .gr file (DIMACS-like format for graphs) and returns the
    number of vertices and an adjacency list representation of the graph.
    Vertices in the file are assumed to be 1-indexed.

    Args:
        filepath (str): The path to the .gr file.

    Returns:
        tuple: (num_vertices, adj_list)
               num_vertices (int): The total number of vertices in the graph.
               adj_list (dict): A dictionary where keys are vertex IDs (int, 1-indexed)
                                and values are sets of their neighbors (int, 1-indexed).
    """
    adj_list = {}
    num_vertices = 0
    problem_line_parsed = False # Flag to ensure 'p ds ...' is the first relevant line

    with open(filepath, 'r') as f:
        for line_num, line_content in enumerate(f, 1):
            line_content = line_content.strip()

            # Skip empty lines or comments
            if not line_content or line_content.startswith('c'):
                continue
            
            parts = line_content.split()
            
            # Parse the problem descriptor line (e.g., "p ds 7 9")
            if parts[0] == 'p':
                if problem_line_parsed:
                    print(f"Warning: Multiple 'p' lines found. Using the first one. (Line {line_num})", file=sys.stderr)
                    continue
                if len(parts) == 4 and parts[1] == 'ds':
                    try:
                        num_vertices = int(parts[2])
                        # num_edges = int(parts[3]) # Stored but not directly used by this solver's core logic
                        # Initialize adjacency list for all vertices
                        for i in range(1, num_vertices + 1):
                            adj_list[i] = set()
                        problem_line_parsed = True
                    except ValueError:
                        print(f"Error: Malformed problem line '{line_content}' (Line {line_num}). Could not parse n, m.", file=sys.stderr)
                        sys.exit(1) # Critical error, cannot proceed
                else:
                    print(f"Error: Invalid problem descriptor line '{line_content}' (Line {line_num}). Expected 'p ds n m'.", file=sys.stderr)
                    sys.exit(1) # Critical error
            
            # Parse edge lines (e.g., "1 2")
            elif problem_line_parsed: # Only parse edges after 'p ds ...'
                if len(parts) == 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        if 1 <= u <= num_vertices and 1 <= v <= num_vertices:
                            adj_list[u].add(v)
                            adj_list[v].add(u) # Graph is undirected
                        else:
                            print(f"Warning: Edge '{line_content}' (Line {line_num}) contains out-of-range vertex for n={num_vertices}. Skipping.", file=sys.stderr)
                    except ValueError:
                        print(f"Warning: Could not parse edge line '{line_content}' (Line {line_num}). Skipping.", file=sys.stderr)
                else:
                     print(f"Warning: Malformed edge line '{line_content}' (Line {line_num}) after problem descriptor. Skipping.", file=sys.stderr)
            else:
                # Line encountered before 'p ds ...' that is not a comment or empty
                print(f"Warning: Unexpected line '{line_content}' (Line {line_num}) before 'p ds ...' line. Skipping.", file=sys.stderr)

    if not problem_line_parsed and num_vertices == 0: # Check if 'p ds' line was ever found
        print("Error: No 'p ds n m' problem line found in the input file.", file=sys.stderr)
        sys.exit(1)
            
    return num_vertices, adj_list

def get_msc_problem_setup(num_vertices, adj_list):
    """
    Transforms the graph problem into a Minimum Set Cover (MSC) problem setup.
    The universe U for MSC is the set of all graph vertices.
    The collection of sets S_collection for MSC consists of the closed neighborhoods N[v]
    for every vertex v in the graph.

    Args:
        num_vertices (int): The total number of vertices.
        adj_list (dict): Adjacency list of the graph.

    Returns:
        list: A list of tuples, where each tuple is (frozenset_N_v, original_vertex_v_id).
              frozenset_N_v is the closed neighborhood of vertex v.
              original_vertex_v_id is the ID of vertex v (1-indexed).
    """
    s_tuples_initial = []  # List of (frozenset_N[v], original_vertex_v_id)
    if num_vertices == 0:
        return s_tuples_initial

    for v_id in range(1, num_vertices + 1):
        # Closed neighborhood N[v] = N(v) U {v}
        closed_neighborhood = adj_list.get(v_id, set()) | {v_id} # Use .get for safety, though parser should ensure keys exist
        s_tuples_initial.append((frozenset(closed_neighborhood), v_id))
        
    return s_tuples_initial

def del_operation(S_chosen_fset, current_S_tuples):
    """
    Implements the 'del(S, S_collection)' operation from the paper.
    Given a chosen set S_chosen_fset and the current collection of sets (current_S_tuples),
    it returns a new collection where each set R from current_S_tuples is replaced by
    R - S_chosen_fset. Empty sets resulting from this operation are discarded.

    Args:
        S_chosen_fset (frozenset): The set chosen to be IN the set cover.
        current_S_tuples (list): The current list of (frozenset, original_vertex_id) tuples.

    Returns:
        list: A new list of (frozenset, original_vertex_id) tuples for the subproblem.
    """
    new_S_tuples = []
    for R_fset, R_orig_v_id in current_S_tuples:
        # Note: If R_fset is S_chosen_fset itself, Z will be empty.
        Z = R_fset - S_chosen_fset 
        if Z:  # Only add if the resulting set Z is not empty
            new_S_tuples.append((Z, R_orig_v_id))
    return new_S_tuples

def msc_recursive(current_S_tuples, U_target_to_cover):
    """
    Recursively solves the Minimum Set Cover problem based on the algorithm
    described in Chapter 2 of the "Measure and Conquer: Domination – A Case Study" paper.

    Args:
        current_S_tuples (list): A list of (frozenset, original_vertex_id) pairs,
                                 representing the current available sets S_collection.
        U_target_to_cover (set): A set of elements (vertex IDs) that still need to be covered.

    Returns:
        tuple: (cost, chosen_ds_vertex_ids)
               cost (float): The minimum number of sets required to cover U_target_to_cover.
                             Can be float('inf') if not possible.
               chosen_ds_vertex_ids (set): A set of original vertex IDs corresponding to
                                           the chosen sets forming the minimum cover.
    """
    global memo

    # --- Memoization Check ---
    # Create a canonical key for the current subproblem state.
    # This ensures that identical subproblems are recognized even if the order of sets
    # or elements within sets differs in the input arguments.
    # The key consists of:
    #   1. A sorted tuple of sorted tuples representing the frozensets in current_S_tuples.
    #   2. A frozenset of the elements in U_target_to_cover.
    canonical_S_frozensets_list = sorted([tuple(sorted(list(s_tuple[0]))) for s_tuple in current_S_tuples])
    memo_key = (tuple(canonical_S_frozensets_list), frozenset(U_target_to_cover))

    if memo_key in memo:
        return memo[memo_key]

    # --- Base Cases (Paper's Figure 1, Line 2 implicitly) ---
    # Case 1: All target elements are already covered.
    if not U_target_to_cover:
        return (0, set())

    # Case 2: No sets are available to cover the remaining target elements.
    if not current_S_tuples:
        return (float('inf'), set()) # Cannot cover the target

    # --- Simplification Rule 1 (Paper's Figure 1, Line 3 & Lemma 1.1) ---
    # If set S_i is a (proper) subset of S_j, then S_i can be removed without affecting optimality.
    # If S_i is identical to S_j, one of them can be removed.
    # This loop applies the rule once if a suitable pair is found, then recurses.
    for i in range(len(current_S_tuples)):
        S_fset_i, _ = current_S_tuples[i]
        for j in range(len(current_S_tuples)):
            if i == j:
                continue
            S_fset_j, _ = current_S_tuples[j]

            # Case 1a: S_i is a proper subset of S_j. Remove S_i.
            if S_fset_i.issubset(S_fset_j) and S_fset_i != S_fset_j:
                # Recurse with S_i removed from the collection.
                next_S_tuples_rule1 = [t for idx, t in enumerate(current_S_tuples) if idx != i]
                result = msc_recursive(next_S_tuples_rule1, U_target_to_cover)
                memo[memo_key] = result
                return result
            
            # Case 1b: S_i and S_j are identical. Remove one (e.g., S_j if i < j for determinism).
            if S_fset_i == S_fset_j and i < j: # Ensure we only remove one copy
                # Recurse with S_j removed.
                next_S_tuples_rule1 = [t for idx, t in enumerate(current_S_tuples) if idx != j]
                result = msc_recursive(next_S_tuples_rule1, U_target_to_cover)
                memo[memo_key] = result
                return result

    # --- Simplification Rule 2 (Paper's Figure 1, Line 4 & Lemma 1.2) ---
    # If an element u_target in U_target_to_cover is contained in exactly one set S_unique
    # from current_S_tuples, then S_unique must be part of any minimum set cover.
    S_unique_tuple_for_rule2 = None
    # Iterate through target elements (sorted for deterministic behavior if multiple unique elements exist)
    for u_target_elem in sorted(list(U_target_to_cover)): 
        sets_covering_u_target = []
        for s_fset, s_orig_v_id in current_S_tuples:
            if u_target_elem in s_fset:
                sets_covering_u_target.append((s_fset, s_orig_v_id))
        
        if len(sets_covering_u_target) == 1:
            S_unique_tuple_for_rule2 = sets_covering_u_target[0]
            break # Found an element covered by a unique set, apply the rule.
    
    if S_unique_tuple_for_rule2:
        S_unique_fset, S_unique_orig_v_id = S_unique_tuple_for_rule2
        
        # Prepare for recursive call:
        # New collection of sets: apply del_operation with S_unique_fset.
        # New target universe: remove elements covered by S_unique_fset.
        next_S_tuples_rule2 = del_operation(S_unique_fset, current_S_tuples)
        next_U_target_rule2 = U_target_to_cover - S_unique_fset
        
        cost_recursive, chosen_ds_recursive = msc_recursive(next_S_tuples_rule2, next_U_target_rule2)
        
        final_result_rule2 = (float('inf'), set())
        if cost_recursive != float('inf'):
            # Cost is 1 (for S_unique) + cost from the recursive call.
            # Chosen vertices include S_unique's original vertex and those from recursion.
            final_result_rule2 = (1 + cost_recursive, chosen_ds_recursive | {S_unique_orig_v_id})
        
        memo[memo_key] = final_result_rule2
        return final_result_rule2

    # --- Branching (Paper's Figure 1, Lines 5-7) ---
    # If no simplification rules applied, select a set S_chosen and branch.
    # Line 5: "take S in S_collection of maximum cardinality".
    # S_chosen_tuple is (frozenset_N[v], original_vertex_v_id).
    # The paper's algorithm doesn't specify tie-breaking for max cardinality; Python's max() is deterministic.
    S_chosen_tuple = max(current_S_tuples, key=lambda t: len(t[0]))
    S_chosen_fset, S_chosen_orig_v_id = S_chosen_tuple
    
    # Line 6 (poly-msc for |S|=2) is omitted in this implementation for simplicity.
    # The general branching logic of Line 7 will handle these cases.

    # Branch 1: S_chosen is NOT included in the set cover (OUT branch).
    # Create the next collection by removing S_chosen_tuple.
    S_collection_for_out_branch = list(current_S_tuples) # Make a mutable copy
    try:
        S_collection_for_out_branch.remove(S_chosen_tuple) # Removes the first occurrence
    except ValueError:
        # This should ideally not happen if S_chosen_tuple was selected from current_S_tuples.
        # Fallback if S_chosen_tuple was somehow not in the list (defensive).
        S_collection_for_out_branch = [t for t in current_S_tuples if t != S_chosen_tuple]

    cost_out, chosen_ds_out = msc_recursive(S_collection_for_out_branch, U_target_to_cover)

    # Branch 2: S_chosen IS included in the set cover (IN branch).
    # Cost is 1 (for choosing S_chosen) + cost of covering remaining elements.
    U_target_for_in_branch = U_target_to_cover - S_chosen_fset
    S_collection_for_in_branch = del_operation(S_chosen_fset, current_S_tuples)
    
    cost_in_recursive, chosen_ds_in_recursive = msc_recursive(S_collection_for_in_branch, U_target_for_in_branch)

    cost_in = float('inf')
    chosen_ds_in = set()
    if cost_in_recursive != float('inf'):
        cost_in = 1 + cost_in_recursive
        chosen_ds_in = chosen_ds_in_recursive | {S_chosen_orig_v_id}

    # --- Combine results from IN and OUT branches ---
    final_result = (float('inf'), set())
    if cost_in < cost_out:
        final_result = (cost_in, chosen_ds_in)
    elif cost_out < cost_in:
        final_result = (cost_out, chosen_ds_out)
    else: # Costs are equal (this includes both being float('inf'))
        if cost_in == float('inf'): # If both are infinite, no solution through this path
            final_result = (float('inf'), set())
        else: # Costs are equal and finite. Apply a tie-breaking rule.
              # Prefer solution with fewer vertices in the dominating set.
              # If still tied, prefer the 'IN' branch solution for determinism.
            if len(chosen_ds_in) < len(chosen_ds_out):
                 final_result = (cost_in, chosen_ds_in)
            elif len(chosen_ds_out) < len(chosen_ds_in):
                 final_result = (cost_out, chosen_ds_out)
            else: # Same cost, same number of chosen vertices
                 final_result = (cost_in, chosen_ds_in) # Arbitrary choice: prefer IN

    memo[memo_key] = final_result
    return final_result

def solve_mds_from_file(gr_filepath):
    """
    Main orchestrator function to solve the Minimum Dominating Set problem
    for a graph specified in a .gr file.
    """
    global memo
    memo = {} # Reset memoization cache for each new problem instance

    num_vertices, adj_list = parse_gr_file(gr_filepath)

    if num_vertices == 0: # Handle empty graph case
        return 0, []

    # Convert graph to MSC problem setup
    s_tuples_initial = get_msc_problem_setup(num_vertices, adj_list)
    U_target_initial = set(range(1, num_vertices + 1)) # All vertices must be covered

    min_cost, chosen_vertices_for_ds = msc_recursive(s_tuples_initial, U_target_initial)

    if min_cost == float('inf'):
        # This indicates an issue, as a dominating set (e.g., all vertices) should always exist for non-empty graphs.
        print(f"Warning: MSC algorithm returned infinite cost for graph in {gr_filepath}. This is unexpected for valid graphs.", file=sys.stderr)
        return float('inf'), [] # Propagate error/no solution state

    return min_cost, sorted(list(chosen_vertices_for_ds)) # Return sorted list of vertices

def write_output_file(output_filepath, solution_size, solution_vertices):
    """
    Writes the solution (size and list of vertices) to the output file
    in the specified DIMACS-like format.
    """
    with open(output_filepath, 'w') as f:
        f.write(f"c Solution for Dominating Set problem generated by Python script\n")
        if solution_size == float('inf') or not isinstance(solution_size, (int, float)) : # Check for error states
            f.write("c No valid solution found or error during computation.\n")
            f.write("0\n") # Convention for no solution or error
        else:
            f.write(f"{int(solution_size)}\n")
            for vertex in solution_vertices:
                f.write(f"{vertex}\n")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python your_script_name.py <input_gr_file> <output_solution_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Attempt to set a higher recursion limit. This is often necessary for
    # deep recursive algorithms like this one on non-trivial graphs.
    try:
        temp_n, _ = parse_gr_file(input_file) # Quick pre-parse to get n
        if temp_n > 0:
            # Adjust multiplier as needed; N sets could be chosen in worst case for depth
            new_limit = max(sys.getrecursionlimit(), temp_n * 3 + 1000) 
            sys.setrecursionlimit(new_limit)
        else: # For empty or very small graphs, a moderate default is fine.
            sys.setrecursionlimit(max(sys.getrecursionlimit(), 2000))
    except Exception as e_parse_limit:
        print(f"Warning: Could not pre-parse for dynamic recursion limit ({e_parse_limit}). Using a fixed high limit.", file=sys.stderr)
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000)) # Fallback high limit

    print(f"Processing graph from: {input_file} with recursion limit: {sys.getrecursionlimit()}")
    try:
        size, ds_vertices = solve_mds_from_file(input_file)
        
        if size == float('inf'):
            print("Error: No valid dominating set could be found by the algorithm.", file=sys.stderr)
            write_output_file(output_file, 0, []) # Output "0 solution" for error
        else:
            print(f"Minimum Dominating Set Size: {int(size)}")
            print(f"Vertices in Dominating Set: {ds_vertices}")
            write_output_file(output_file, int(size), ds_vertices)
            print(f"Solution written to: {output_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except RecursionError:
        print(f"CRITICAL Error: Python's recursion depth limit was reached.", file=sys.stderr)
        print(f"The graph from '{input_file}' is likely too large or complex for this exact recursive algorithm", file=sys.stderr)
        print(f"with the current recursion limit ({sys.getrecursionlimit()}).", file=sys.stderr)
        print(f"Consider using a smaller graph, an iterative version of the algorithm, or further increasing the limit if system resources allow (though this may lead to stack overflow).", file=sys.stderr)
        try:
            write_output_file(output_file, 0, []) # Output "0 solution" for error
            print(f"Empty solution written to {output_file} due to recursion error.", file=sys.stderr)
        except Exception as e_write:
            print(f"Additionally, failed to write empty solution file: {e_write}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        try:
            write_output_file(output_file, 0, []) # Output "0 solution" for error
            print(f"Empty solution written to {output_file} due to unexpected error.", file=sys.stderr)
        except Exception as e_write:
            print(f"Additionally, failed to write empty solution file: {e_write}", file=sys.stderr)
        sys.exit(1)

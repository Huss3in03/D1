import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power

def nodes_degree(A):
    """
    Function returning the degree of a node.
    Parameters:
    A : Adjacency matrix (assumed symmetric).
    """
    degree = np.sum(A, axis=0)
    return degree

def path_length(A, i, j):
    """
    Function returning the minimum path length between two nodes.
    Parameters:
    A : Adjacency matrix (assumed symmetric).
    i, j : Nodes indices.
    """
    Lij = -1
    
    if A[i, j] > 0:
        Lij = 1
    else:
        N = np.size(A[0, :])
        P = np.zeros([N, N]) + A
        n = 1
        running = True
        while running:
            P = np.matmul(P, A)
            n += 1
            if P[i, j] > 0:
                Lij = n           
            if (n > N) or (Lij > 0):
                running = False   
    return Lij

def matrix_path_length(A):
    """
    Function returning a matrix L of minimum path length between nodes.
    Parameters:
    A : Adjacency matrix (assumed symmetric).
    """
    N = np.size(A[0, :])
    L = np.zeros([N, N]) - 1 
    
    for i in range(N):
        for j in range(i + 1, N):
            L[i, j] = path_length(A, i, j)
            L[j, i] = L[i, j]
    return L

def clustering_coefficient(A):
    """
    Function returning the clustering coefficient of a graph.
    Parameters:
    A : Adjacency matrix (assumed symmetric).
    """
    K = nodes_degree(A)
    N = np.size(K)

    # Avoid division by zero if degree is 0 or 1
    denom = np.sum(K * (K - 1))
    if denom == 0:
        return 0.0

    C_n = np.sum(np.diagonal(matrix_power(A, 3)))
    C_d = denom
    
    C = C_n / C_d
    return C

def watts_strogatz_sw(n, c, p):
    """
    Function generating a Watts-Strogatz small-world model.
    
    Parameters:
    n : Number of nodes.
    c : Number of connected nearest neigbours. Must be even.
    p : Probability that each existing edge is randomly rewired.
    """
    A = np.zeros([n, n])    
    P = np.random.rand(n, n)    
    A_rewired = np.zeros([n, n])    
    
    c_half = int(c / 2)
    
    # 1. Construct Regular Ring Lattice
    for i in range(n):
        for j in range(i + 1, i + 1 + c_half):
            # Use modulo to wrap around the circle
            A[i, j % n] = 1
            A[j % n, i] = 1

    # 2. Random Rewiring
    # Note: The lecture implementation iterates and rewires.
    # Ideally, we verify we don't rewire to self or duplicate edges,
    # but we follow the lecture's logic structure closely.
    for i in range(n):
        for j in range(i + 1, i + 1 + c_half):
            if P[i, j % n] < p:
                # Rewire to a random node k
                # Ensure k != i and no self-loops (simple random choice)
                # In a strict implementation, we also check if connection exists.
                k = (np.random.randint(n - 1) + 1) + i
                A_rewired[i, k % n] = 1
                A_rewired[k % n, i] = 1
            else:
                # Keep original connection
                A_rewired[i, j % n] = 1
                A_rewired[j % n, i] = 1

    # Coordinates for circular plotting
    x = np.cos(np.arange(n) / n * 2 * np.pi)
    y = np.sin(np.arange(n) / n * 2 * np.pi) 
    
    return A_rewired, x, y

def run_part_1():
    print("\n--- PART 1: n=20, c=4 ---")
    n = 20
    c = 4
    
    # Configuration: (p_value, number_of_graphs_to_generate)
    configs = [
        (0.0, 1),
        (0.2, 2),
        (0.4, 2)
    ]
    
    # Prepare figure for plotting the graphs
    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Part 1: Watts-Strogatz Networks (n={n}, c={c})")
    
    plot_index = 1
    
    # Header for the metrics table
    print(f"{'Graph ID':<10} | {'p':<5} | {'Avg Path Length':<15} | {'Diameter':<8} | {'Clustering Coeff':<15}")
    print("-" * 65)
    
    for p, num_graphs in configs:
        for i in range(num_graphs):
            # A) Generate Graph
            A_WS, x, y = watts_strogatz_sw(n, c, p)
            
            # Plotting
            plt.subplot(2, 3, plot_index)
            # Draw edges
            for r in range(n):
                for c_idx in range(r + 1, n):
                    if A_WS[r, c_idx] > 0:
                        plt.plot([x[r], x[c_idx]], [y[r], y[c_idx]], '-', color='gray', linewidth=0.8)
            # Draw nodes
            plt.plot(x, y, '.', markersize=15, color='blue')
            plt.title(f"Graph {plot_index}: p={p}")
            plt.axis('equal')
            plt.axis('off')
            
            # B) Metrics: Path Length and Diameter
            L = matrix_path_length(A_WS)
            
            # Calculate Average Path Length
            # Sum upper triangle of L, divide by number of pairs N*(N-1)/2
            sum_paths = 0
            count_paths = 0
            for r in range(n):
                for c_idx in range(r + 1, n):
                    if L[r, c_idx] > 0: # Check if path exists
                        sum_paths += L[r, c_idx]
                        count_paths += 1
            
            if count_paths > 0:
                avg_path_len = sum_paths / count_paths
                diameter = np.max(L)
            else:
                avg_path_len = float('inf')
                diameter = float('inf')
                
            # C) Metric: Clustering Coefficient
            cc = clustering_coefficient(A_WS)
            
            # Print metrics
            print(f"{plot_index:<10} | {p:<5} | {avg_path_len:<15.4f} | {diameter:<8.0f} | {cc:<15.4f}")
            
            plot_index += 1
            
    plt.tight_layout()
    plt.show()

def run_part_2():
    print("\n--- PART 2: n=100, c=6 ---")
    n = 100
    c = 6
    p_values = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    num_independent_runs = 3
    
    avg_lengths_means = []
    avg_lengths_stds = []
    
    print(f"Calculating average path length l(p) for p values: {p_values}")
    
    for p in p_values:
        lengths_for_current_p = []
        
        for run in range(num_independent_runs):
            # Generate graph
            A_WS, _, _ = watts_strogatz_sw(n, c, p)
            
            # Calculate path length matrix
            L = matrix_path_length(A_WS)
            
            # Calculate average path length for this specific graph
            sum_paths = 0
            count_paths = 0
            for r in range(n):
                for c_idx in range(r + 1, n):
                    # We assume graph is connected, but good to check L > 0
                    if L[r, c_idx] > 0:
                        sum_paths += L[r, c_idx]
                        count_paths += 1
            
            if count_paths > 0:
                l_avg = sum_paths / count_paths
                lengths_for_current_p.append(l_avg)
            
        # Calculate mean and std of the 3 runs
        mean_l = np.mean(lengths_for_current_p)
        std_l = np.std(lengths_for_current_p)
        
        avg_lengths_means.append(mean_l)
        avg_lengths_stds.append(std_l)
        
        print(f"p = {p:<8} : l(p) = {mean_l:.4f} +/- {std_l:.4f}")
        
    # Plotting l(p) with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(p_values, avg_lengths_means, yerr=avg_lengths_stds, 
                 fmt='-o', capsize=5, color='red', ecolor='black', label='Simulated Data')
    
    plt.xscale('log')
    plt.xlabel('Rewiring Probability p')
    plt.ylabel('Average Path Length l(p)')
    plt.title(f'Small World Effect: Path Length vs Rewiring Probability\n(n={n}, c={c})')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.show()

    # E) Comment on the graph
    print("\n(E) COMMENT ON THE GRAPH:")
    print("-" * 30)
    print("The graph illustrates the 'Small-World' phenomenon.")
    print("1. For small p (near 0), the network is regular. The path length is long (linear with n).")
    print("2. As p increases (around 10^-2 to 10^-1), there is a sharp drop in the average path length.")
    print("   This is because the few rewired 'shortcuts' connect distant parts of the ring, drastically reducing separation.")
    print("3. For p approaching 1, the network becomes random, maintaining a small average path length.")
    print("   The Small-World regime is specifically where path length is low (like a random graph) but clustering")
    print("   (not plotted here, but implied) remains high (like a regular graph).")

if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(42)
    
    run_part_1()
    run_part_2()
import numpy as np
import matplotlib.pyplot as plt

def diffuse_spread_recover_hetero(x, y, status, d, beta_arr, gamma_arr, L):
    """
    Function performing the diffusion step, the infection step, and the 
    recovery step happening in one turn for a population of agents.
    
    Supports heterogeneous beta and gamma (arrays) for each agent.
    """
    
    N = np.size(x)
    
    # --- Diffusion step ---
    diffuse = np.random.rand(N)
    move = np.random.randint(4, size=N)
    
    # Vectorized movement logic
    # 0: left, 1: up, 2: right, 3: down
    # Only move if random check < diffusion probability d
    moving_mask = diffuse < d
    
    x[moving_mask & (move == 0)] -= 1
    y[moving_mask & (move == 1)] -= 1
    x[moving_mask & (move == 2)] += 1
    y[moving_mask & (move == 3)] += 1
                
    # Enforce periodic boundary conditions
    x = x % L
    y = y % L

    # --- Spreading disease step ---
    # Identify currently infected agents
    infected = np.where(status == 1)[0]
    
    # Shuffle to ensure random interaction order
    np.random.shuffle(infected)

    for i in infected:
        # Find agents in the exact same cell (x, y)
        # Note: For N=2000 this simple search is fast enough. 
        # For much larger N, a spatial hash/grid map would be preferred.
        in_same_cell = np.where((x == x[i]) & (y == y[i]))[0]
        
        for j in in_same_cell:
            if status[j] == 0: # If the neighbor is Susceptible
                # Use beta of the SUSCEPTIBLE agent (beta_arr[j])
                # This represents the agent's specific genetic susceptibility
                if np.random.rand() < beta_arr[j]:
                    status[j] = 1
        
    # --- Recover step ---
    # Iterate over currently infected agents
    current_infected = np.where(status == 1)[0]
    for i in current_infected:
        # Use gamma of the INFECTED agent (gamma_arr[i])
        # This represents the agent's specific recovery rate
        if np.random.rand() < gamma_arr[i]:
            status[i] = 2 # Recovered
    
    return x, y, status

def run_simulation_set(N, L, d, beta_arr, gamma_arr, I0, repetitions, title):
    """
    Runs the simulation multiple times and plots all trajectories on one figure.
    """
    plt.figure(figsize=(10, 6))
    
    colors = {'S': 'blue', 'I': 'red', 'R': 'green'}
    
    print(f"Starting simulation set: {title}")
    
    for r in range(repetitions):
        # Initialize positions
        x = np.random.randint(L, size=N)
        y = np.random.randint(L, size=N)
        
        # Initialize status: 0=S, 1=I, 2=R
        status = np.zeros(N)
        
        # Randomly choose initial infected
        initial_infected_indices = np.random.choice(N, I0, replace=False)
        status[initial_infected_indices] = 1
        
        S, I, R_counts = [], [], []
        
        # Record initial state
        S.append(np.sum(status == 0))
        I.append(np.sum(status == 1))
        R_counts.append(np.sum(status == 2))
        
        running = True
        step = 0
        
        while running:
            x, y, status = diffuse_spread_recover_hetero(x, y, status, d, beta_arr, gamma_arr, L)
            
            curr_S = np.sum(status == 0)
            curr_I = np.sum(status == 1)
            curr_R = np.sum(status == 2)
            
            S.append(curr_S)
            I.append(curr_I)
            R_counts.append(curr_R)
            
            step += 1
            # Stop if no infected agents remain
            if curr_I == 0:
                running = False
        
        t = np.arange(len(S))
        
        # Plot lines with transparency (alpha) to show variation between runs
        # Only add label for the first run to avoid cluttered legend
        lbl_s = 'S (Susceptible)' if r == 0 else ""
        lbl_i = 'I (Infected)' if r == 0 else ""
        lbl_r = 'R (Recovered)' if r == 0 else ""
        
        plt.plot(t, S, color=colors['S'], alpha=0.4, label=lbl_s)
        plt.plot(t, I, color=colors['I'], alpha=0.4, label=lbl_i)
        plt.plot(t, R_counts, color=colors['R'], alpha=0.4, label=lbl_r)

    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Number of Agents')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- Global Simulation Parameters ---
    N = 2000          # Total agents
    L = 100           # Lattice side length
    d = 0.9           # Diffusion probability
    I0 = 10           # Initial infected
    repetitions = 6   # Number of repetitions per scenario

    # ==========================================
    # PART 1 (A)
    # All individuals are Type 1
    # beta = 0.8, gamma = 0.02
    # ==========================================
    beta_val_1 = 0.8
    gamma_val_1 = 0.02

    beta_arr_A = np.full(N, beta_val_1)
    gamma_arr_A = np.full(N, gamma_val_1)

    run_simulation_set(N, L, d, beta_arr_A, gamma_arr_A, I0, repetitions, 
                       title="(A) Type 1 Population (beta=0.8, gamma=0.02)")

    # ==========================================
    # PART 1 (B)
    # All individuals are Type 2
    # beta = 0.1, gamma = 0.01
    # ==========================================
    beta_val_2 = 0.1
    gamma_val_2 = 0.01

    beta_arr_B = np.full(N, beta_val_2)
    gamma_arr_B = np.full(N, gamma_val_2)

    run_simulation_set(N, L, d, beta_arr_B, gamma_arr_B, I0, repetitions, 
                       title="(B) Type 2 Population (beta=0.1, gamma=0.01)")

    # ==========================================
    # PART 1 (C) - Comments
    # ==========================================
    print("\n--- (C) Comparison of (A) and (B) ---")
    print("In Case A (Type 1), the disease is highly infectious (beta=0.8). "
          "You will observe a sharp, high peak in infections (I) that occurs quickly. "
          "The number of Susceptible (S) agents drops to nearly zero.")
    print("In Case B (Type 2), the infection probability is much lower (beta=0.1). "
          "The epidemic curve is flatter and wider. The peak number of infected is significantly "
          "lower than in Case A, and the disease may take longer to die out due to the lower "
          "recovery rate (gamma=0.01), or it might die out very early if the density isn't high enough.")

    # ==========================================
    # PART 2 (D)
    # Mixed Population
    # 1000 Type 1 agents + 1000 Type 2 agents
    # ==========================================
    
    # Create arrays
    b1 = np.full(1000, beta_val_1)
    g1 = np.full(1000, gamma_val_1)

    b2 = np.full(1000, beta_val_2)
    g2 = np.full(1000, gamma_val_2)

    # Concatenate to form population of 2000
    beta_arr_D = np.concatenate((b1, b2))
    gamma_arr_D = np.concatenate((g1, g2))

    run_simulation_set(N, L, d, beta_arr_D, gamma_arr_D, I0, repetitions, 
                       title="(D) Mixed Population (50% Type 1, 50% Type 2)")

    # ==========================================
    # PART 2 (E) - Comments
    # ==========================================
    print("\n--- (E) Comparison of Mixed Population (D) vs Part 1 ---")
    print("In the Mixed Population (D), the Type 1 agents (high beta) act as 'super-spreaders' "
          "or fuel for the fire, causing an initial rapid spike in infections similar to Case A.")
    print("However, the Type 2 agents (low beta, low gamma) contract the disease later or slower. "
          "Because Type 2 agents have a lower recovery rate (gamma=0.01 vs 0.02), they stay infected "
          "longer, dragging out the 'tail' of the epidemic.")
    print("This often results in a curve that looks like a hybrid: a sharp rise (driven by Type 1) "
          "followed by a slower decline (driven by Type 2).")
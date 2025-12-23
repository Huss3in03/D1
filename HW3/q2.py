import numpy as np
import matplotlib.pyplot as plt
import math
from functools import reduce

def replicas(x, y, L):
    """
    Function to generate replicas of a single particle for PBC.
    """    
    xr = np.zeros(9)
    yr = np.zeros(9)

    for i in range(3):
        for j in range(3):
            xr[3 * i + j] = x + (j - 1) * L
            yr[3 * i + j] = y + (i - 1) * L
    return xr, yr

def pbc(x, y, L):
    """
    Function to enforce periodic boundary conditions on the positions.
    """   
    outside_left = np.where(x < - L / 2)[0]
    x[outside_left] = x[outside_left] + L

    outside_right = np.where(x > L / 2)[0]
    x[outside_right] = x[outside_right] - L

    outside_up = np.where(y > L / 2)[0]
    y[outside_up] = y[outside_up] - L

    outside_down = np.where(y < - L / 2)[0]
    y[outside_down] = y[outside_down] + L
    return x, y

def calculate_intensity(x, y, I0, r0, L, r_c):
    """
    Function to calculate the intensity seen by each particle.
    Sums contributions from all other particles.
    """
    N = np.size(x)
    I_particle = np.zeros(N) 
    
    # Preselect particles closer than r_c to boundaries to check replicas
    replicas_needed = reduce( 
        np.union1d, (
            np.where(y + r_c > L / 2)[0], 
            np.where(y - r_c < - L / 2)[0],
            np.where(x + r_c > L / 2)[0],
            np.where(x - r_c < - L / 2)[0]
        )
    )

    for j in range(N - 1):   
        # Check if replicas are needed
        if np.size(np.where(replicas_needed == j)[0]):
            xr, yr = replicas(x[j], y[j], L)
            for nr in range(9):
                # Calculate distance to all subsequent particles
                dist2 = (x[j + 1:] - xr[nr]) ** 2 + (y[j + 1:] - yr[nr]) ** 2 
                nn = np.where(dist2 <= r_c ** 2)[0] + j + 1
                
                if np.size(nn) > 0:
                    nn = nn.astype(int)
                    dx = x[nn] - xr[nr]
                    dy = y[nn] - yr[nr]
                    d2 = dx ** 2 + dy ** 2
                    I = I0 * np.exp(- d2 / r0 ** 2)
                    
                    I_particle[j] += np.sum(I)
                    I_particle[nn] += I
        else:
            dist2 = (x[j + 1:] - x[j]) ** 2 + (y[j + 1:] - y[j]) ** 2 
            nn = np.where(dist2 <= r_c ** 2)[0] + j + 1
        
            if np.size(nn) > 0:
                nn = nn.astype(int)
                dx = x[nn] - x[j]
                dy = y[nn] - y[j]
                d2 = dx ** 2 + dy ** 2
                I = I0 * np.exp(- d2 / r0 ** 2)
                
                I_particle[j] += np.sum(I)
                I_particle[nn] += I
                    
    return I_particle

def run_mixed_simulation(N, L, idx_neg, idx_pos, delta_steps, duration, dt, params):
    """
    Runs the simulation for N robots with mixed delays.
    
    Parameters:
    - idx_neg: list of indices for robots with negative delay
    - idx_pos: list of indices for robots with positive delay
    - delta_steps: integer delay in steps (used for both pos and neg magnitude)
    """
    
    # Unpack physics parameters
    v0 = params['v0']
    v_inf = params['v_inf']
    Ic = params['Ic']
    I0 = params['I0']
    r0 = params['r0']
    tau = params['tau']
    
    r_c = 4 * r0 
    n_steps = math.ceil(duration / dt)
    
    # Initialize State
    x = (np.random.rand(N) - 0.5) * L
    y = (np.random.rand(N) - 0.5) * L
    phi = 2 * (np.random.rand(N) - 0.5) * np.pi
    
    # Trajectory Storage
    traj_x = np.zeros((n_steps, N))
    traj_y = np.zeros((n_steps, N))
    
    c_noise_phi = np.sqrt(2 / tau * dt)
    
    # History Buffer
    # Size needed is delta_steps + 1 to look back 'delta_steps' into the past
    buf_size = delta_steps + 1
    I_history = np.zeros((buf_size, N))
    
    # Initialize history with current state
    I_start = calculate_intensity(x, y, I0, r0, L, r_c)
    for k in range(buf_size):
        I_history[k, :] = I_start

    t_fit = np.arange(buf_size) * dt
    
    print(f"Running simulation: {N} robots ({len(idx_neg)} neg delay, {len(idx_pos)} pos delay) for {duration}s...")
    
    for i in range(n_steps):
        # Store positions
        traj_x[i, :] = x
        traj_y[i, :] = y
        
        # Calculate current intensity
        I_curr = calculate_intensity(x, y, I0, r0, L, r_c)
        
        # Update rolling buffer
        I_history = np.roll(I_history, -1, axis=0)
        I_history[-1, :] = I_curr
        
        # Calculate Effective Intensity I_eff based on delay type
        I_eff = np.zeros(N)
        
        # 1. Positive Delay: v(t) = v(I(t - delta))
        # Access oldest element in buffer (index 0)
        if len(idx_pos) > 0:
            I_eff[idx_pos] = I_history[0, idx_pos]
            
        # 2. Negative Delay: v(t) = v(I(t + delta))
        # Linearly extrapolate into the future
        if len(idx_neg) > 0:
            for r_idx in idx_neg:
                # Fit line to history
                poly = np.polyfit(t_fit, I_history[:, r_idx], 1)
                dI_dt = poly[0]
                # Extrapolate: Current + slope * time_delta
                pred = I_curr[r_idx] + (delta_steps * dt) * dI_dt
                if pred < 0: pred = 0
                I_eff[r_idx] = pred

        # Update Velocity
        v = v_inf + (v0 - v_inf) * np.exp(- I_eff / Ic)
        
        # Update Position (Euler method)
        rn = np.random.normal(0, 1, N)
        x = x + v * dt * np.cos(phi)
        y = y + v * dt * np.sin(phi)
        phi = phi + c_noise_phi * rn
        
        # Apply PBC
        x, y = pbc(x, y, L)
        
    return traj_x, traj_y

def plot_results(traj_x, traj_y, special_idx, L, title_prefix, sub_label):
    """
    Generates the requested plots: 
    1. Initial/Final Configuration + Trajectory
    2. 2D Exploration Histogram
    """
    
    # --- Plot A/D: Config & Trajectory ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Config Plot
    # Start positions
    axes[0].scatter(traj_x[0, :], traj_y[0, :], c='gray', alpha=0.5, label='Start (All)')
    axes[0].scatter(traj_x[0, special_idx], traj_y[0, special_idx], c='green', s=100, marker='x', label='Start (Subject)')
    
    # Final positions
    others = [i for i in range(traj_x.shape[1]) if i != special_idx]
    axes[0].scatter(traj_x[-1, others], traj_y[-1, others], c='blue', alpha=0.7, label='End (Majority)')
    axes[0].scatter(traj_x[-1, special_idx], traj_y[-1, special_idx], c='red', s=100, edgecolors='k', label='End (Subject)')
    
    axes[0].set_xlim(-L/2, L/2)
    axes[0].set_ylim(-L/2, L/2)
    axes[0].set_aspect('equal')
    axes[0].set_title(f"{title_prefix}: Initial & Final Config")
    axes[0].legend()
    axes[0].grid(True, linestyle=':')

    # Trajectory Plot (Subject Robot)
    axes[1].plot(traj_x[:, special_idx], traj_y[:, special_idx], 'k-', linewidth=0.5, alpha=0.6)
    axes[1].set_xlim(-L/2, L/2)
    axes[1].set_ylim(-L/2, L/2)
    axes[1].set_aspect('equal')
    axes[1].set_title(f"{title_prefix}: Trajectory of Subject")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

    # --- Plot B/E: 2D Histogram ---
    plt.figure(figsize=(7, 6))
    plt.hist2d(traj_x[:, special_idx], traj_y[:, special_idx], bins=50, 
               range=[[-L/2, L/2], [-L/2, L/2]], cmap='inferno')
    cb = plt.colorbar()
    cb.set_label('Time steps spent')
    plt.title(f"{title_prefix}: Exploration Density ({sub_label})")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    # Parameters
    params = {
        'v0': 0.1,      # m/s
        'v_inf': 0.01,  # m/s
        'Ic': 0.1,      # W/m^2
        'I0': 1.0,      # W/m^2
        'r0': 0.3,      # m
        'tau': 1.0      # s
    }
    L = 3.0             # m
    dt = 0.05           # s
    T_tot = 1800        # s (Total duration)
    N_total = 10
    delta_steps = 5     # Delay magnitude in steps (5 * dt)

    print("\n--- STARTING PART 1 ---")
    idx_neg_p1 = [0]                # Robot 0 is Negative
    idx_pos_p1 = list(range(1, 10)) # Robots 1-9 are Positive
    
    tx1, ty1 = run_mixed_simulation(N_total, L, idx_neg_p1, idx_pos_p1, delta_steps, T_tot, dt, params)
    
    plot_results(tx1, ty1, idx_neg_p1[0], L, "Part 1 (Neg Subject)", "Negative Delay Robot")
    
    print("Part 1 Comments:")
    print("The majority (Positive Delay) form dense clusters due to delayed reaction to high intensity.")
    print("The subject (Negative Delay) anticipates intensity rises and avoids deep trapping, exploring more freely.")

    print("\n--- STARTING PART 2 ---")
    idx_pos_p2 = [0]                # Robot 0 is Positive
    idx_neg_p2 = list(range(1, 10)) # Robots 1-9 are Negative
    
    tx2, ty2 = run_mixed_simulation(N_total, L, idx_neg_p2, idx_pos_p2, delta_steps, T_tot, dt, params)
    
    plot_results(tx2, ty2, idx_pos_p2[0], L, "Part 2 (Pos Subject)", "Positive Delay Robot")

    print("Part 2 Comments:")
    print("The majority (Negative Delay) disperse effectively, creating a dynamic, fluctuating field.")
    print("The subject (Positive Delay) attempts to cluster but finds no stable wells, resulting in a 'frustrated' walk.")
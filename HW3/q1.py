import math
import numpy as np 
from matplotlib import pyplot as plt


# Physical Constants
kB = 1.380e-23       # Boltzmann constant [J/K]
T = 300              # Temperature [K]
eta = 1e-3           # Viscosity [N s/m^2]
R = 1e-6             # Radius [m]

# Derived Constants
gamma = 6 * np.pi * R * eta       # Translational drag
gammaR = 8 * np.pi * R**3 * eta   # Rotational drag

D = kB * T / gamma                # Translational Diffusion [m^2/s]
DR = kB * T / gammaR              # Rotational Diffusion [1/s]
tau_R = 1 / DR                    # Orientation relaxation time [s]

# Simulation Parameters
v = 50e-6            # Self-propulsion speed [m/s]
dt = 20e-3           # Time step [s]
T_tot = 10000        # Total time [s] (at least 10000s as requested)

# Angular velocities [0, pi/2, pi, 3pi/2] rad/s
omega_list = np.array([0, 0.5 * np.pi, np.pi, 1.5 * np.pi]) 


def evolution_chiral_ABP(x0, y0, phi0, v, omega, D, DR, dt, noise_array):
    """
    Simulates the trajectory of a chiral active Brownian particle using pre-generated noise.
    """
    # Coefficients for the finite difference solution.
    c_noise_x = np.sqrt(2 * D * dt)
    c_noise_y = np.sqrt(2 * D * dt)
    c_noise_phi = np.sqrt(2 * DR * dt)

    N = noise_array.shape[1] + 1  # Total steps based on noise array size

    x = np.zeros(N)
    y = np.zeros(N)
    phi = np.zeros(N)
    
    x[0] = x0
    y[0] = y0
    phi[0] = phi0

    for i in range(N - 1):
        # Update Position
        x[i + 1] = x[i] + v * dt * np.cos(phi[i]) + c_noise_x * noise_array[0, i]
        y[i + 1] = y[i] + v * dt * np.sin(phi[i]) + c_noise_y * noise_array[1, i]
        
        # Update Orientation (includes chiral term omega)
        phi[i + 1] = phi[i] + omega * dt + c_noise_phi * noise_array[2, i]

    return x, y, phi

def MSD_calc(x, y, n_delays):
    """
    Calculates Mean Squared Displacement for specific time delays (vectorized).
    """
    L = np.size(n_delays)
    msd = np.zeros(L)
    
    for i in range(L):
        n = n_delays[i]
        # Calculate displacements for all intervals of length n simultaneously
        dx = x[n:] - x[:-n]
        dy = y[n:] - y[:-n]
        msd[i] = np.mean(dx**2 + dy**2)

    return msd


if __name__ == "__main__":
    print("--- PART 1: SIMULATION ---")
    print(f"Translational Diffusion D: {D:.3e} m^2/s")
    print(f"Rotational Diffusion DR:   {DR:.3f} 1/s")
    print(f"Relaxation time tau_R:     {tau_R:.3f} s")
    print("-" * 30)

    # 1. Generate Noise ONCE (to be shared across all omegas)
    N_steps = math.ceil(T_tot / dt)
    np.random.seed(42) # Fixed seed for reproducibility
    noise_common = np.random.normal(0, 1, size=(3, N_steps - 1))

    # 2. Run Simulations
    trajectories = {}
    x0, y0, phi0 = 0, 0, 0
    
    print("Simulating trajectories...")
    for w in omega_list:
        x, y, phi = evolution_chiral_ABP(x0, y0, phi0, v, w, D, DR, dt, noise_common)
        trajectories[w] = {'x': x, 'y': y, 'phi': phi}
    print("Simulation complete.\n")

    
    time_array = dt * np.arange(N_steps)
    limit_time = 2 * tau_R
    limit_idx = int(limit_time / dt)

    # Plot (A): XY Trajectories (Initial section)
    plt.figure(figsize=(8, 8))
    for w in omega_list:
        plt.plot(trajectories[w]['x'][:limit_idx], 
                 trajectories[w]['y'][:limit_idx], 
                 '-', linewidth=2, label=f"$\omega = {w/np.pi:.1f}\pi$ rad/s")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'(A) XY Trajectories (t <= {limit_time:.1f}s)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot (B): Orientation vs Time
    plt.figure(figsize=(10, 5))
    for w in omega_list:
        plt.plot(time_array[:limit_idx], 
                 trajectories[w]['phi'][:limit_idx], 
                 '-', linewidth=2, label=f"$\omega = {w/np.pi:.1f}\pi$")
    plt.xlabel('Time [s]')
    plt.ylabel('Orientation $\phi$ [rad]')
    plt.title('(B) Orientation vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(">> Answers to (A) & (B):")
    print(f"   (A) The value of tau_R is {tau_R:.4f} seconds.")
    print("   (B) Implications: The graph shows that non-zero omega adds a linear drift")
    print("       to the orientation angle. The fluctuations are identical because the")
    print("       same noise was used. The slope corresponds to the angular velocity.\n")

    
    print("--- PART 2: MSD CALCULATION ---")
    # Generate logarithmically spaced delays
    min_pow = np.log10(1)
    max_pow = np.log10(N_steps // 10) 
    n_delays = np.unique(np.logspace(min_pow, max_pow, num=50).astype(int))
    t_delay = n_delays * dt
    
    MSD_results = {}
    
    for w in omega_list:
        MSD_results[w] = MSD_calc(trajectories[w]['x'], trajectories[w]['y'], n_delays)

    # Plot (C): MSD Log-Log
    plt.figure(figsize=(8, 6))
    
    # Guides
    plt.loglog(t_delay, 1e-11 * t_delay**2, 'k--', alpha=0.5, label='~ t^2 (Ballistic)')
    plt.loglog(t_delay, 1e-9 * t_delay, 'k:', alpha=0.5, label='~ t (Diffusive)')
    
    for w in omega_list:
        plt.loglog(t_delay, MSD_results[w], '.-', linewidth=1.5, label=f"$\omega = {w/np.pi:.1f}\pi$")
    
    plt.xlabel('Time lag $\Delta t$ [s]')
    plt.ylabel('MSD [m$^2$]')
    plt.title('(C) Mean Squared Displacement')
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)

    print(">> Answers to (C) & (D):")
    print("   (D) Ballistic behavior (slope 2) is observed at short times (t << tau_R).")
    print("       Diffusive behavior (slope 1) is observed at long times (t >> tau_R).")
    print("       Peculiarity: For omega > 0, oscillations appear in the MSD curve")
    print("       corresponding to the rotational period of the particle.")

    # Show all plots at once
    plt.show()
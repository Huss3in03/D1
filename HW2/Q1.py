import numpy as np
import matplotlib.pyplot as plt
import math, time

kB = 1.380e-23        # J/K
T = 300.0             # K
eta = 1e-3            # Ns/m^2
R = 1e-6              # m
sigma = R / 2.0
dt = 2e-2             # s
Ttot = 3 * 3600.0     # 3 hours
Nsteps = int(np.ceil(Ttot / dt))

gamma = 6.0 * np.pi * eta * R
D = kB * T / gamma

print(f"Nsteps = {Nsteps}")
print(f"gamma = {gamma:.3e}, D = {D:.3e}")

def potential(x, U):
    return -U * np.exp(-x**2 / (2.0 * sigma**2))

def force(x, U):
    return -(U / sigma**2) * x * np.exp(-x**2 / (2.0 * sigma**2))

# U_n = n kBT
ns = np.arange(1, 9)
Uns = ns * kB * T

xgrid = np.linspace(-6*sigma, 6*sigma, 1000)

plt.figure(figsize=(6,4))
for n, U in zip(ns, Uns):
    plt.plot(xgrid*1e6, potential(xgrid, U), label=f"n={n}")
plt.xlabel("x (µm)")
plt.ylabel("U(x) (J)")
plt.title("Gaussian trap potentials U_n(x)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
for n, U in zip(ns, Uns):
    plt.plot(xgrid*1e6, force(xgrid, U), label=f"n={n}")
plt.xlabel("x (µm)")
plt.ylabel("F(x) (N)")
plt.title("Forces F(x)")
plt.legend()
plt.tight_layout()
plt.show()

escape_threshold = 5.0 * sigma
Nrep = 5

escape_times = []  # store triplets (n, rep, escape_time or NaN)

rng = np.random.default_rng(seed=42)
start_time = time.time()

for n, U in zip(ns, Uns):
    for rep in range(Nrep):

        x = 0.0
        escaped = False

        for i in range(Nsteps):
            Fx = force(x, U)
            drift = (Fx / gamma) * dt
            noise = np.sqrt(2.0 * D * dt) * rng.standard_normal()
            x = x + drift + noise

            if abs(x) > escape_threshold:
                escape_times.append((int(n), rep+1, (i+1)*dt))
                escaped = True
                break

        if not escaped:
            escape_times.append((int(n), rep+1, float('nan')))

    print(f"Completed n={n}")

print(f"Finished all simulations in {time.time()-start_time:.1f} s")

sample_trajs = {}
rng = np.random.default_rng(seed=123)

for n, U in zip(ns, Uns):
    trajs = []
    for rep in range(Nrep):
        x = 0.0
        xs = [x]
        ts = [0.0]

        for i in range(Nsteps):
            Fx = force(x, U)
            drift = (Fx / gamma) * dt
            noise = np.sqrt(2.0 * D * dt) * rng.standard_normal()
            x = x + drift + noise

            xs.append(x)
            ts.append((i+1)*dt)

            if abs(x) > escape_threshold:
                break

        trajs.append((np.array(ts), np.array(xs)))
    sample_trajs[int(n)] = trajs

# Plot B sample trajectories
fig, axes = plt.subplots(4,2, figsize=(10,12), sharex=True)
axes = axes.flatten()

for idx, n in enumerate(ns):
    ax = axes[idx]
    for (ts, xs) in sample_trajs[n]:
        ax.plot(ts/3600.0, xs*1e6)
    ax.axhline(escape_threshold*1e6, ls='--')
    ax.axhline(-escape_threshold*1e6, ls='--')
    ax.set_title(f"n={n}")
    if idx >= 6:
        ax.set_xlabel("time (hours)")
    ax.set_ylabel("x (µm)")

plt.tight_layout()
plt.show()

summary_n = []
summary_mean = []
summary_std = []

for n in ns:
    vals = []
    for (nval, rep, t) in escape_times:
        if nval == n and not math.isnan(t):
            vals.append(t)

    if len(vals) == 0:
        mean = float('nan')
        std = float('nan')
    else:
        mean = np.mean(vals)
        std = np.std(vals)

    summary_n.append(n)
    summary_mean.append(mean)
    summary_std.append(std)

# Plot C: escape times vs n
plt.figure(figsize=(6,4))
plt.errorbar(summary_n,
             np.array(summary_mean)/3600.0,
             yerr=np.array(summary_std)/3600.0,
             marker='o')
plt.yscale("log")
plt.xlabel("n")
plt.ylabel("Mean escape time (hours, log scale)")
plt.title("Escape time vs trap depth")
plt.grid(True, which="both")
plt.tight_layout()
plt.show()

# Print summary to terminal
print("\nESCAPE TIME SUMMARY TABLE")
print(" n |  mean escape time (s)  |  std escape time (s)")
print("---------------------------------------------------")
for n, m, s in zip(summary_n, summary_mean, summary_std):
    m_str = f"{m:.4g}" if not math.isnan(m) else "NaN"
    s_str = f"{s:.4g}" if not math.isnan(s) else "NaN"
    print(f" {n} | {m_str:>22} | {s_str:>21}")
print("---------------------------------------------------\n")


# Parameters for the two traps
x_left = -0.9 * R
x_right = 0.9 * R
sigma_left = R / 2
sigma_right = R / 2
U_left = 8 * kB * T
U_right = 8 * kB * T

# Potential of two traps
def U_double(x):
    U1 = -U_left * np.exp(-(x - x_left)**2 / (2*sigma_left**2))
    U2 = -U_right * np.exp(-(x - x_right)**2 / (2*sigma_right**2))
    return U1 + U2

# Corresponding forces
def F_double(x):
    F1 = -(U_left / sigma_left**2) * (x - x_left) * np.exp(-(x - x_left)**2 / (2*sigma_left**2))
    F2 = -(U_right / sigma_right**2) * (x - x_right) * np.exp(-(x - x_right)**2 / (2*sigma_right**2))
    return F1 + F2

xgrid = np.linspace(-3*R, 3*R, 1500)

plt.figure(figsize=(6,4))
plt.plot(xgrid*1e6, U_double(xgrid))
plt.xlabel("x (µm)")
plt.ylabel("U(x) (J)")
plt.title("Double Gaussian Potential")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(xgrid*1e6, F_double(xgrid))
plt.xlabel("x (µm)")
plt.ylabel("F(x) (N)")
plt.title("Force for Double Trap")
plt.tight_layout()
plt.show()

Nrep2 = 3
traj2 = []

rng2 = np.random.default_rng(seed=999)

for rep in range(Nrep2):

    x = 0.0
    xs = [x]
    ts = [0.0]

    for i in range(Nsteps):
        Fx = F_double(x)
        drift = (Fx / gamma) * dt
        noise = np.sqrt(2.0 * D * dt) * rng2.standard_normal()
        x = x + drift + noise

        xs.append(x)
        ts.append((i+1)*dt)

    traj2.append((np.array(ts), np.array(xs)))

# Plot trajectories
plt.figure(figsize=(8,5))
for (ts, xs) in traj2:
    plt.plot(ts/3600.0, xs*1e6)

plt.axhline(x_left*1e6, ls='--', color='gray')
plt.axhline(x_right*1e6, ls='--', color='gray')
plt.xlabel("Time (hours)")
plt.ylabel("x (µm)")
plt.title("Trajectories in Double Gaussian Trap")
plt.tight_layout()
plt.show()

# Detect transitions: left well -> right well and reverse
def count_transitions(ts, xs):
    # Determine wells:
    # left = x < 0, right = x > 0
    states = np.sign(xs)   # -1 = left, +1 = right
    changes = np.where(np.diff(states) != 0)[0]
    return len(changes)

transition_counts = []
transition_rates = []

for (ts, xs) in traj2:
    Ntr = count_transitions(ts, xs)
    transition_counts.append(Ntr)
    transition_rates.append(Ntr / (ts[-1]))  # per second

print("\nTRANSITION FREQUENCY RESULTS")
for i, rate in enumerate(transition_rates):
    print(f" Trajectory {i+1}: {rate:.3e} transitions/s")

print("\nAverage transition rate:", np.mean(transition_rates), "transitions/s")


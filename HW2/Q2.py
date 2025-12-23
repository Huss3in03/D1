import numpy as np
import matplotlib.pyplot as plt

kB = 1.380e-23      # Boltzmann const (J/K)
T0 = 300.0          # temperature (K)
eta = 1e-3          # viscosity (N s/m^2)
R = 1e-6            # particle radius (m)

gamma = 6 * np.pi * eta * R
k = 1e-6            # trap stiffness (N/m)
D = kB * T0 / gamma # diffusion coefficient

dt = 2e-3           # time step (s)


def generate_trajectory(T_total):
    N = int(np.round(T_total / dt))
    x = np.zeros(N)

    # thermalize
    tau = gamma / k
    ntherm = int(np.ceil(5 * tau / dt))
    ntherm = min(ntherm, 200000)

    x0 = 0.0
    for ii in range(ntherm):
        if ii % 50000 == 0:
            print(f"Thermalizing... {ii}/{ntherm}")
        x0 = x0 - (k/gamma) * x0 * dt + np.sqrt(2*D*dt)*np.random.normal()

    x[0] = x0

    for n in range(N - 1):
        if n % 50000 == 0:
            print(f"Generating trajectory... {n}/{N}")
        x[n+1] = x[n] - (k/gamma)*x[n]*dt + np.sqrt(2*D*dt)*np.random.normal()

    return x


def tMSD(x, y):
    N = len(x)
    msd = np.zeros(N)
    for n in range(N):
        if n % 2000 == 0:
            print(f"Computing tMSD lag {n}/{N}")
        dx = x[n:] - x[:N-n]
        dy = y[n:] - y[:N-n]
        msd[n] = np.mean(dx**2 + dy**2)
    return msd


def ensemble_MSD_efficient(T_total, Ntraj):
    N = int(np.round(T_total / dt))

    tau = gamma / k
    ntherm = int(np.ceil(5 * tau / dt))
    ntherm = min(ntherm, 200000)

    print(f"Thermalizing {Ntraj} trajectories...")
    x0s = np.zeros(Ntraj)
    y0s = np.zeros(Ntraj)
    for i in range(Ntraj):
        if i % 10 == 0:
            print(f"  Thermalizing traj {i}/{Ntraj}")
        x = 0.0
        y = 0.0
        for _ in range(ntherm):
            x = x - (k/gamma)*x*dt + np.sqrt(2*D*dt)*np.random.normal()
            y = y - (k/gamma)*y*dt + np.sqrt(2*D*dt)*np.random.normal()
        x0s[i] = x
        y0s[i] = y

    eMSD = np.zeros(N)

    chunk = 2000
    x = x0s.copy()
    y = y0s.copy()

    print("Starting ensemble MSD simulation...")
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        L = end - start

        print(f"  Processing chunk {start}–{end} / {N}")

        noise_x = np.random.normal(size=(Ntraj, L))
        noise_y = np.random.normal(size=(Ntraj, L))

        for j in range(L):
            x = x - (k/gamma)*x*dt + np.sqrt(2*D*dt)*noise_x[:, j]
            y = y - (k/gamma)*y*dt + np.sqrt(2*D*dt)*noise_y[:, j]
            eMSD[start + j] = np.mean((x - x0s)**2 + (y - y0s)**2)

    print("ensemble_MSD_efficient finished.\n")
    return eMSD

T1 = 60.0
Ntraj1 = 10

print("\n=== Running Case (a) ===")

print("Generating single trajectory...")
x1 = generate_trajectory(T1)
y1 = generate_trajectory(T1)

print("Computing tMSD (a)...")
tmsd1 = tMSD(x1, y1)

print("Computing eMSD (a)...")
emsd1 = ensemble_MSD_efficient(T1, Ntraj1)

t1 = dt * np.arange(len(tmsd1))

plt.figure()
plt.loglog(t1[1:], tmsd1[1:], '-', label='tMSD')
plt.loglog(t1[1:], emsd1[1:], '--', label='eMSD')
plt.xlabel("t (s)")
plt.ylabel("MSD (m^2)")
plt.title("Case (a): T = 60 s, Ntraj = 10")
plt.legend()
plt.show()


T2 = 360.0
Ntraj2 = 100

print("\n=== Running Case (b) ===")

print("Generating single trajectory...")
x2 = generate_trajectory(T2)
y2 = generate_trajectory(T2)

print("Computing tMSD (b)...")
tmsd2 = tMSD(x2, y2)

print("Computing eMSD (b)...")
emsd2 = ensemble_MSD_efficient(T2, Ntraj2)

t2 = dt * np.arange(len(tmsd2))

plt.figure()
plt.loglog(t2[1:], tmsd2[1:], '-', label='tMSD')
plt.loglog(t2[1:], emsd2[1:], '--', label='eMSD')
plt.xlabel("t (s)")
plt.ylabel("MSD (m^2)")
plt.title("Case (b): T = 360 s, Ntraj = 100")
plt.legend()
plt.show()

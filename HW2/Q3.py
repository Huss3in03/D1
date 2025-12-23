import numpy as np
import matplotlib.pyplot as plt
from math import pi

dt = 1.0        # time step
sigma0 = 1.0    # base standard deviation of noise
L = 100.0       # length of the box
x0 = 0.0        # initial particle position
lam = 10.0      # parameter controlling σ(x) slope
t0 = 100        # reference time


# Position-dependent standard deviation
def sigma(x):
    return sigma0 * (2/pi * np.arctan(2*lam*x/L) + 1)

# Derivative of σ(x) with respect to x
def dsigdx(x):
    return (4*sigma0*lam)/(pi*L) * 1/(1 + (2*lam*x/L)**2)

# Spurious drift s(x) = σ(x) * dσ/dx
def s_of_x(x):
    return sigma(x) * dsigdx(x)

xs = np.linspace(-L/2, L/2, 1001)  # x values across the box
s_vals = s_of_x(xs)                 # compute spurious drift at each x

plt.figure(figsize=(8,4))
plt.plot(xs, s_vals)                # plot s(x)
plt.xlabel("x")
plt.ylabel("s(x) = σ dσ/dx")
plt.title("Spurious drift $s(x)$ across the domain")
plt.grid(True)
plt.show()


def simulate(alpha, ws, x0):
    N = len(ws)
    x = np.zeros(N+1)  # store trajectory
    x[0] = x0          # set initial position

    for n in range(N):
        xn = x[n]

        # spurious drift term
        drift_spurious = alpha * sigma(xn) * dsigdx(xn) * dt
        # random noise term
        noise_term = sigma(xn) * np.sqrt(dt) * ws[n]

        xnext = xn + drift_spurious + noise_term

        # Reflective boundaries
        if xnext > L/2:
            xnext = L/2 - (xnext - L/2)
        if xnext < -L/2:
            xnext = -L/2 - (xnext + L/2)

        x[n+1] = xnext

    return x

Ttot = 2 * t0                  # total simulation time
N = int(Ttot / dt)             # number of time steps

rng = np.random.default_rng(0)
ws = rng.normal(0, 1, N)       # random numbers for noise

x_ito  = simulate(alpha=0, ws=ws, x0=x0)    # Ito trajectory
x_anti = simulate(alpha=1, ws=ws, x0=x0)    # anti-Ito trajectory

# Plot trajectories
plt.figure(figsize=(10,4))
plt.plot(x_ito, label="Ito (α=0)")
plt.plot(x_anti, label="anti-Ito (α=1)")
plt.xlabel("time step")
plt.ylabel("x(t)")
plt.grid(True)
plt.legend()
plt.title("Single realization: Ito vs anti-Ito")
plt.show()

# Spurious drift time series along anti-Ito trajectory
s_time = s_of_x(x_anti[:-1])   # exclude last step since drift uses current x

plt.figure(figsize=(10,3))
plt.plot(s_time)
plt.xlabel("time step")
plt.ylabel("s(x(t))")
plt.grid(True)
plt.title("Spurious drift along anti-Ito trajectory")
plt.show()

Nr = 100000     # number of independent trajectories
Tmax = 100*t0   # total time steps for simulation

rng2 = np.random.default_rng(1)
ws_all = rng2.normal(0, 1, size=(Nr, Tmax))  # generate noise for all particles

X = np.zeros((Nr, Tmax+1))  # store all trajectories
X[:,0] = x0                 # initial positions

for n in range(Tmax):
    xn = X[:,n]

    # spurious drift and random noise
    drift = sigma(xn) * dsigdx(xn)
    noise = sigma(xn) * ws_all[:,n]

    xnext = xn + drift + noise

    # Reflective boundaries
    over = xnext > L/2
    xnext[over] = L/2 - (xnext[over] - L/2)
    under = xnext < -L/2
    xnext[under] = -L/2 - (xnext[under] + L/2)

    X[:,n+1] = xnext

# Times to examine for histograms
times = [t0, 5*t0, 10*t0, 25*t0, 50*t0, 100*t0]

plt.figure(figsize=(10,10))
for i,t in enumerate(times):
    plt.subplot(len(times),1,i+1)
    plt.hist(X[:,t], bins=60)       # histogram of positions at time t
    plt.xlim(-L/2, L/2)
    plt.ylabel(f"t = {t}")
plt.xlabel("x")
plt.suptitle("Final position distribution (anti-Ito α=1)")
plt.tight_layout()
plt.show()

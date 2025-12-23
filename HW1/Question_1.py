import numpy as np
import math
import time
from tkinter import *
import matplotlib.pyplot as plt

# Parameters for the Lennard-Jones gas.
m = 1  # Mass (units of m0).
sigma = 1  # Size (units of sigma0).
eps = 1  # Energy (unit of epsilon0).
v0 = 1  # Initial speed (units of v0 = sqrt((2 * epsilon0) / m0)).

# Parameters for the simulation.
N_particles = 100  # Number of particles.

dt = 0.001   # Time step (units of t0 = sigma * sqrt(m0 /(2 * epsilon0))).
T_tot = 10

L = 16 * sigma  # Box size (units of sigma0).
x_min, x_max, y_min, y_max = -L/2, L/2, -L/2, L/2

cutoff_radius = 5 * sigma  # Cutoff_radius for neighbours list.

# Generate initial positions on a grid and orientations at random.
x0, y0 = np.meshgrid(
    np.linspace(- L / 2, L / 2, int(np.sqrt(N_particles))),
    np.linspace(- L / 2, L / 2, int(np.sqrt(N_particles))),
)
x0 = x0.flatten()[:N_particles]
y0 = y0.flatten()[:N_particles]
phi0 = (2 * np.random.rand(N_particles) - 1) * np.pi

# Initialize the neighbour list.
def list_neighbours(x, y, N_particles, cutoff_radius):
    '''Prepare a neigbours list for each particle.'''
    neighbours = []
    neighbour_number = []
    for j in range(N_particles):
        distances = np.sqrt((x - x[j]) ** 2 + (y - y[j]) ** 2)
        neighbor_indices = np.where(distances <= cutoff_radius)
        neighbours.append(neighbor_indices)
        neighbour_number.append(len(neighbor_indices))
    return neighbours, neighbour_number

neighbours, neighbour_number = list_neighbours(x0, y0, N_particles, cutoff_radius)

# Initialize the variables for the leapfrog algorithm.
# Current time step.
x = x0
y = y0
x_half = np.zeros(N_particles)
y_half = np.zeros(N_particles)
v = v0
phi = phi0
vx = v0 * np.cos(phi0)
vy = v0 * np.sin(phi0)

# Next time step.
nx = np.zeros(N_particles)
ny = np.zeros(N_particles)
nv = np.zeros(N_particles)
nphi = np.zeros(N_particles)
nvx = np.zeros(N_particles)
nvy = np.zeros(N_particles)

def total_force_cutoff(x, y, N_particles, sigma, epsilon, neighbours):
    '''
    Calculate the total force on each particle due to the interaction with a 
    neighbours list with the particles interacting through a Lennard-Jones 
    potential.
    '''
    Fx = np.zeros(N_particles)
    Fy = np.zeros(N_particles)
    for i in range(N_particles):
        for j in list(neighbours[i][0]):
            if i != j:
                r2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                r = np.sqrt(r2)
                ka2 = sigma ** 2 / r2
                
                # Force on i due to j.
                F = 24 * epsilon / r * (2 * ka2 ** 6 - ka2 ** 3)  # Modulus.
                
                Fx[i] += F * (x[i] - x[j]) / r
                Fy[i] += F * (y[i] - y[j]) / r
    return Fx, Fy

window_size = 600

tk = Tk()
tk.geometry(f'{window_size + 20}x{window_size + 20}')
tk.configure(background='#000000')

canvas = Canvas(tk, background='#ECECEC')  # Generate animation window 
tk.attributes('-topmost', 0)
canvas.place(x=10, y=10, height=window_size, width=window_size)

specialparticle = canvas.create_oval(
    (x[0] - sigma / 2) / L * window_size + window_size / 2, 
    (y[0] - sigma / 2) / L * window_size + window_size / 2,
    (x[0] + sigma / 2) / L * window_size + window_size / 2, 
    (y[0] + sigma / 2) / L * window_size + window_size / 2,
    outline='#000000', 
    fill='#000000',
)

particles = []
for j in range(1, N_particles):
    particles.append(
        canvas.create_oval(
            (x[j] - sigma / 2) / L * window_size + window_size / 2, 
            (y[j] - sigma / 2) / L * window_size + window_size / 2,
            (x[j] + sigma / 2) / L * window_size + window_size / 2, 
            (y[j] + sigma / 2) / L * window_size + window_size / 2,
            outline='#00C0C0', 
            fill='#00C0C0',
        )
    )

center_idx = int(np.argmin(np.sqrt(x**2 + y**2)))  # particle closest to box center
center_x_list = []
center_y_list = []
time_list = []

step = 0

def stop_loop(event):
    global running
    running = False
tk.bind("<Escape>", stop_loop)  # Bind the Escape key to stop the loop.
running = True  # Flag to control the loop.
while running and step < (T_tot/dt) :
    x_half = x + 0.5 * vx * dt      
    y_half = y + 0.5 * vy * dt      

    fx, fy = \
        total_force_cutoff(x_half, y_half, N_particles, sigma, eps, neighbours)
    
    nvx = vx + fx / m * dt
    nvy = vy + fy / m * dt
        
    nx = x_half + 0.5 * nvx * dt
    ny = y_half + 0.5 * nvy * dt       
    
    # Reflecting boundary conditions.
    for j in range(N_particles):
        if nx[j] < x_min:
            nx[j] = x_min + (x_min - nx[j])
            nvx[j] = - nvx[j]

        if nx[j] > x_max:
            nx[j] = x_max - (nx[j] - x_max)
            nvx[j] = - nvx[j]

        if ny[j] < y_min:
            ny[j] = y_min + (y_min - ny[j])
            nvy[j] = - nvy[j]
            
        if ny[j] > y_max:
            ny[j] = y_max - (ny[j] - y_max)
            nvy[j] = - nvy[j]
    
    nv = np.sqrt(nvx ** 2 + nvy ** 2)
    for i in range(N_particles):
        nphi[i] = math.atan2(nvy[i], nvx[i])
    
    # Update neighbour list.
    if step % 10 == 0:
        neighbours, neighbour_number = \
            list_neighbours(nx, ny, N_particles, cutoff_radius)

    # Update variables for next iteration.
    x = nx
    y = ny
    vx = nvx
    vy = nvy
    v = nv
    phi = nphi

    center_x_list.append(x[center_idx])
    center_y_list.append(y[center_idx])
    time_list.append(step * dt)
        
    # Update animation frame.
    if step % 100 == 0:        
        canvas.coords(
            specialparticle,
            (nx[0] - sigma / 2) / L * window_size + window_size / 2,
            (ny[0] - sigma / 2) / L * window_size + window_size / 2,
            (nx[0] + sigma / 2) / L * window_size + window_size / 2,
            (ny[0] + sigma / 2) / L * window_size + window_size / 2,
        )
        for j, particle in enumerate(particles):
            canvas.coords(
                particle,
                (nx[j + 1] - sigma / 2) / L * window_size + window_size / 2,
                (ny[j + 1] - sigma / 2) / L * window_size + window_size / 2,
                (nx[j + 1] + sigma / 2) / L * window_size + window_size / 2,
                (ny[j + 1] + sigma / 2) / L * window_size + window_size / 2,
            )
                    
        tk.title(f'Time {step * dt:.1f} - Iteration {step}')
        tk.update_idletasks()
        tk.update()
        time.sleep(.001)  # Increase to slow down the simulation.    

    step += 1

tk.update_idletasks()
tk.update()
tk.mainloop()  # Release animation handle (close window to finish).

cx = np.array(center_x_list)
cy = np.array(center_y_list)
t  = np.array(time_list)
S = len(t)
if S == 0:
    print("No trajectory points recorded (S=0).")
else:
    # Trajectory plot (x vs y)
    plt.figure(figsize=(5,5))
    plt.plot(cx, cy, marker='.', markersize=2, linestyle='-')
    plt.scatter(cx[0], cy[0], color='green', label='start')
    plt.scatter(cx[-1], cy[-1], color='red', label='end')
    plt.title(f'Center particle trajectory (index {center_idx})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Mean square displacement (MSD)
    max_lag = S // 2
    msd = np.zeros(max_lag)
    pos = np.vstack((cx, cy)).T
    for n in range(1, max_lag):
        diffs = pos[n:] - pos[:-n]
        sqdist = (diffs ** 2).sum(axis=1)
        msd[n] = sqdist.mean()

    lags = np.arange(max_lag) * dt
    plt.figure(figsize=(6,4))
    plt.loglog(lags[1:], msd[1:], marker='o', linestyle='-')
    plt.xlabel('Lag time $\\tau$')
    plt.ylabel('MSD($\\tau$)')
    plt.title(f'Mean square displacement of particle {center_idx} (S={S})')
    plt.grid(True, which='both')
    plt.show()

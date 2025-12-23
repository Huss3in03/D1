import numpy as np
import random
import time
from tkinter import *
import matplotlib.pyplot as plt

N = 100  # Size of the spin lattice.
H = 0  # External field.
J = 1  # Spin-spin coupling.
T = 2.3  # Temperature. Critical temperature ~2.269.

d_half_list = [3, 5, 7, 10]
s1 = +1
s2 = +1
steps = 6000 

sl = 2 * np.random.randint(2, size=(N, N)) - 1

N_up = np.sum(sl + 1) / 2
N_down = N * N - N_up

print(f"Spin lattice created:  N_up={N_up}  N_down={N_down}")

def neighboring_spins(i_list, j_list, sl):
    Ni, Nj = sl.shape  
    i_r = i_list  
    j_r = list(map(lambda x:(x + 1) % Nj, j_list))   
    i_l = i_list  
    j_l = list(map(lambda x:(x - 1) % Nj, j_list))   
    i_u = list(map(lambda x:(x - 1) % Ni, i_list))  
    j_u = j_list  
    i_d = list(map(lambda x:(x + 1) % Ni, i_list)) 
    j_d = j_list   
    sl_u = sl[i_u, j_u]
    sl_d = sl[i_d, j_d]
    sl_l = sl[i_l, j_l]
    sl_r = sl[i_r, j_r]
    return sl_u, sl_d, sl_l, sl_r

def energies_spins(i_list, j_list, sl, H, J):
    sl_u, sl_d, sl_l, sl_r = neighboring_spins(i_list, j_list, sl)
    sl_s = sl_u + sl_d + sl_l + sl_r 
    E_u = - H - J * sl_s
    E_d =   H + J * sl_s 
    return E_u, E_d

def probabilities_spins(i_list, j_list, sl, H, J, T):
    E_u, E_d = energies_spins(i_list, j_list, sl, H, J)
    Ei = np.array([E_u, E_d])
    Z = np.sum(np.exp(- Ei / T), axis=0)
    pi = 1 / np.array([Z, Z]) * np.exp(- Ei / T)
    return pi, Z       

def energy_tot(sl, J):
    up = np.roll(sl, -1, axis=0)
    down = np.roll(sl, 1, axis=0)
    right = np.roll(sl, -1, axis=1)
    left = np.roll(sl, 1, axis=1)
    ssum = up + down + right + left
    E = - J * np.sum(sl * ssum) / 2.0
    return E / (sl.shape[0] * sl.shape[1])

results = []

for d_half in d_half_list:
    print(f"\n\nRunning simulation for d_half = {d_half}\n")

    sl = 2 * np.random.randint(2, size=(N, N)) - 1

    row1 = N//2 - d_half
    row2 = N//2 + d_half
    sl[row1, :] = s1
    sl[row2, :] = s2
    frozen = {row1, row2}

    f = 0.05
    N_skip = 10
    window_size = 600

    tk = Tk()
    tk.geometry(f'{window_size + 20}x{window_size + 20}')
    tk.configure(background='#000000')

    canvas = Canvas(tk, background='#ECECEC')
    tk.attributes('-topmost', 0)
    canvas.place(x=10, y=10, height=window_size, width=window_size)

    Nspins = np.size(sl)
    Ni, Nj = sl.shape

    S = int(np.ceil(Nspins * f))
    step = 0
    e_list = []

    def stop_loop(event):
        global running
        running = False
    tk.bind("<Escape>", stop_loop)
    running = True

    while running and step < steps:
        ns = random.sample(range(Nspins), S)
        i_list = list(map(lambda x: x % Ni, ns))
        j_list = list(map(lambda x: x // Ni, ns))
        keep = [k for k in range(len(i_list)) if i_list[k] not in frozen]
        i_try = [i_list[k] for k in keep]
        j_try = [j_list[k] for k in keep]

        if len(i_try) > 0:
            pi, Z = probabilities_spins(i_try, j_try, sl, H, J, T)
            rn = np.random.rand(len(i_try))
            for p in range(len(i_try)):
                sl[i_try[p], j_try[p]] = 1 if rn[p] <= pi[0, p] else -1

        e_list.append(energy_tot(sl, J))

        if step % N_skip == 0:
            canvas.delete('all')
            for i in range(Ni):
                for j in range(Nj):
                    spin_color = '#FFFFFF' if sl[i,j] == 1 else '#000000'
                    canvas.create_rectangle(
                        j / Nj * window_size,
                        i / Ni * window_size,
                        (j + 1) / Nj * window_size,
                        (i + 1) / Ni * window_size,
                        outline='',
                        fill=spin_color,
                    )
            tk.title(f'd_half={d_half}   Iteration {step}')
            tk.update_idletasks()
            tk.update()
            time.sleep(0.1)

        step += 1

    tk.update_idletasks()
    tk.update()
    tk.mainloop() 

    tail = min(1000, len(e_list)//5)
    eq = np.mean(e_list[-tail:])
    print(f"Equilibrium energy for d_half={d_half}:  {eq:.4f}")

    results.append((d_half, eq))

    plt.figure(figsize=(6,4))
    plt.plot(e_list, linewidth=1)
    plt.axhline(eq, linestyle='--')
    plt.title(f"Energy vs Step   (d_half = {d_half})")
    plt.xlabel("Monte Carlo Step")
    plt.ylabel("e_tot")
    plt.show()

print("\nSummary of Equilibrium Energies:")
print("d_half\te_tot")
for d_half, eq in results:
    print(f"{d_half}\t{eq:.6f}")

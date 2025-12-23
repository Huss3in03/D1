import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import *

def neighbors_Moore(status):
    n_nn = (
        np.roll(status, 1, axis=0) +  
        np.roll(status, -1, axis=0) +  
        np.roll(status, 1, axis=1) +  
        np.roll(status, -1, axis=1) +  
        np.roll(np.roll(status, 1, axis=0), 1, axis=1) +  
        np.roll(np.roll(status, 1, axis=0), -1, axis=1) +  
        np.roll(np.roll(status, -1, axis=0), 1, axis=1) +  
        np.roll(np.roll(status, -1, axis=0), -1, axis=1)  
    )
    return n_nn

def apply_rule_2d(rule_2d, status):
    Ni, Nj = status.shape  
    next_status = np.zeros([Ni, Nj]) 
    n_nn = neighbors_Moore(status) 
    for i in range(Ni):
        for j in range(Nj):
            next_status[i, j] = rule_2d[int(status[i, j]), int(n_nn[i, j])]
    return next_status

N = 100
p_values = [0.45, 0.48, 0.50, 0.52, 0.55]
V1_list = []

# Majority rule definition:
rule_2d = np.zeros([2, 9])
rule_2d[0, :] = [0, 0, 0, 0, 0, 0, 1, 1, 1]  # vote 0 → becomes 1 if ≥5 neighbors are 1
rule_2d[1, :] = [0, 0, 0, 0, 1, 1, 1, 1, 1]  # vote 1 → stays 1 if ≥4 neighbors are 1

for p in p_values:
    gol = np.random.choice([0, 1], size=[N, N], p=[1-p, p])
    prev = np.zeros_like(gol)
    step = 0

    tk = Tk()
    window_size = 400
    tk.geometry(f'{window_size+20}x{window_size+20}')
    tk.configure(background='#000000')
    canvas = Canvas(tk, background='#ECECEC')
    canvas.place(x=10, y=10, height=window_size, width=window_size)
    tk.title(f'p = {p:.2f}')

    running = True
    def stop_loop(event):
        global running
        running = False
    tk.bind("<Escape>", stop_loop)

    while running:
        gol = apply_rule_2d(rule_2d, gol)
        step += 1

        # Check convergence
        if np.array_equal(gol, prev):
            running = False
        prev = gol.copy()

        if step % 3 == 0:
            canvas.delete('all')
            for i in range(N):
                for j in range(N):
                    color = '#FFFFFF' if gol[i, j] == 1 else '#000000'
                    canvas.create_rectangle(
                        j/N*window_size, i/N*window_size,
                        (j+1)/N*window_size, (i+1)/N*window_size,
                        outline='', fill=color
                    )
            tk.update_idletasks()
            tk.update()
            time.sleep(0.02)

    V1 = np.mean(gol)
    V1_list.append(V1)
    print(f"p={p:.2f}, Final V1={V1:.3f}")
    tk.mainloop()

plt.figure(figsize=(10,4))

plt.plot(p_values, V1_list, 'o-', lw=2)
plt.xlabel('Initial fraction p')
plt.ylabel('Final fraction V₁(p)')
plt.title('Final Fraction of Votes = 1')

plt.tight_layout()
plt.show()

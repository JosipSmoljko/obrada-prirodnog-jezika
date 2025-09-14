import numpy as np
import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from itertools import combinations

# Parametri
GRID_H, GRID_W = 20, 20   # veličina mreže
N_CAMERAS = 5             # broj kamera
MIN_DIST = 5              # minimalna udaljenost (u ćelijama)
SEED = 1                  # za ponovljivost
SMOOTH_ITERS = 5          # broj iteracija zaglađivanja

# Generiranje nasumičnog terena
np.random.seed(SEED)
terrain = np.random.rand(GRID_H, GRID_W)
for _ in range(SMOOTH_ITERS):
    terrain = (terrain
               + np.roll(terrain, 1, axis=0) + np.roll(terrain, -1, axis=0)
               + np.roll(terrain, 1, axis=1) + np.roll(terrain, -1, axis=1)) / 5

# Pomoćne funkcije
def index_to_coord(idx):
    return divmod(idx, GRID_W)

def grid_distance(a, b):
    (i1, j1), (i2, j2) = a, b
    return ((i1 - i2)**2 + (j1 - j2)**2)**0.5

# ---------- ZADATAK 1 ----------
def solve_task1():
    problem = LpProblem("Zadatak1", LpMaximize)
    x = [LpVariable(f"x_{k}", cat="Binary") for k in range(GRID_H * GRID_W)]
    flat = terrain.flatten()

    # ciljna funkcija: maksimiziraj zbroj visina
    problem += lpSum(flat[k] * x[k] for k in range(GRID_H * GRID_W))
    # točno N kamera
    problem += lpSum(x) == N_CAMERAS

    problem.solve()
    return [index_to_coord(k) for k in range(GRID_H * GRID_W) if x[k].value() == 1]

# ---------- ZADATAK 2 ----------
def solve_task2():
    problem = LpProblem("Zadatak2", LpMaximize)
    x = [LpVariable(f"x_{k}", cat="Binary") for k in range(GRID_H * GRID_W)]
    flat = terrain.flatten()

    problem += lpSum(flat[k] * x[k] for k in range(GRID_H * GRID_W))
    problem += lpSum(x) == N_CAMERAS

    # ograničenje minimalne udaljenosti
    for a, b in combinations(range(GRID_H * GRID_W), 2):
        if grid_distance(index_to_coord(a), index_to_coord(b)) < MIN_DIST:
            problem += x[a] + x[b] <= 1

    problem.solve()
    return [index_to_coord(k) for k in range(GRID_H * GRID_W) if x[k].value() == 1]

# ---------- POKRETANJE ----------
print("Pokrećem optimizaciju...")

sol1 = solve_task1()
sol2 = solve_task2()

print("Gotovo!\n")
print("Zadatak 1 (visina) - kamere na:", sol1)
print("Zadatak 2 (min udaljenost) - kamere na:", sol2)

# ---------- PRIKAZ ----------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(terrain, cmap="terrain")
plt.title("Zadatak 1")
plt.scatter([j for i,j in sol1], [i for i,j in sol1],
            c="red", s=150, edgecolors="black")

plt.subplot(1, 2, 2)
plt.imshow(terrain, cmap="terrain")
plt.title("Zadatak 2")
plt.scatter([j for i,j in sol2], [i for i,j in sol2],
            c="blue", s=150, edgecolors="black")

plt.show()

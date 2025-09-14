# lab5_zadatak3.py
# Zadatak 3 — Pareto analiza restorana (ocjena & udaljenost)
# Prije pokretanja instalirajte potrebne biblioteke:
#   pip install pandas matplotlib paretoset

import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset

# Ucitavanje podataka (zamijenite svojim stvarnim podacima ako zelite)
df = pd.read_csv("restaurants.csv")

# Maksimiziramo ocjenu, minimiziramo udaljenost
mask = paretoset(df[["srednja_ocjena", "udaljenost_km"]], sense=["max", "min"])
pareto_df = df[mask]

print("Pareto optimalni restorani:")
print(pareto_df[["naziv", "srednja_ocjena", "udaljenost_km"]])

# Vizualizacija (2D)
plt.figure()
plt.scatter(df["udaljenost_km"], df["srednja_ocjena"], label="Svi restorani")
plt.scatter(pareto_df["udaljenost_km"], pareto_df["srednja_ocjena"], marker="*", s=200, label="Pareto odluka")

for _, row in df.iterrows():
    plt.annotate(row["naziv"], (row["udaljenost_km"], row["srednja_ocjena"]), textcoords="offset points", xytext=(0,5), ha="center")

plt.xlabel("Udaljenost (km) (manje je bolje)")
plt.ylabel("Srednja ocjena (vece je bolje)")
plt.title("Pareto analiza restorana — ocjena & udaljenost")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

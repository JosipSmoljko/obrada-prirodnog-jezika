# lab5_zadatak2.py
# Zadatak 2 — Pareto analiza hotela (cijena & udaljenost)
# Prije pokretanja instalirajte potrebne biblioteke:
#   pip install pandas matplotlib paretoset

import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset

# Ucitavanje podataka
df = pd.read_csv("hotels.csv")

# Pareto maska: minimiziramo cijenu i udaljenost od plaze
mask = paretoset(df, sense=["min", "min"])
pareto_df = df[mask]

print("Pareto optimalni hoteli (indeksi redaka referiraju na hotels.csv):")
print(pareto_df)

# Vizualizacija (2D)
plt.figure()
plt.scatter(df["udaljenost_od_plaze"], df["cijena"], label="Svi hoteli")
plt.scatter(pareto_df["udaljenost_od_plaze"], pareto_df["cijena"], marker="*", s=200, label="Pareto odluka")

# Oznake tocaka indeksom
for i, row in df.iterrows():
    plt.annotate(str(i), (row["udaljenost_od_plaze"], row["cijena"]), textcoords="offset points", xytext=(0,5), ha="center")

plt.xlabel("Udaljenost od plaze (manje je bolje)")
plt.ylabel("Cijena (manje je bolje)")
plt.title("Pareto analiza hotela")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

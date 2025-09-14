# lab5_zadatak4.py
# Zadatak 4 — Pareto analiza hotela (cijena, udaljenost, recenzije)
# Prije pokretanja instalirajte potrebne biblioteke:
#   pip install pandas matplotlib paretoset
# Napomena: Za 3D prikaz koristi se mpl_toolkits.mplot3d

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from paretoset import paretoset

# Ucitavanje podataka
df = pd.read_csv("hotels_with_reviews.csv")

# Pareto maska: min cijena, min udaljenost, max recenzije
mask = paretoset(df, sense=["min", "min", "max"])
pareto_df = df[mask]

print("Pareto optimalni hoteli (3 kriterija):")
print(pareto_df)

# 3D vizualizacija
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(df["udaljenost_od_plaze"], df["cijena"], df["recenzije"], label="Svi hoteli")
ax.scatter(pareto_df["udaljenost_od_plaze"], pareto_df["cijena"], pareto_df["recenzije"], marker="*", s=200, label="Pareto odluka")

for i, row in df.iterrows():
    ax.text(row["udaljenost_od_plaze"], row["cijena"], row["recenzije"], f"#{i}", size=8, zorder=1)

ax.set_xlabel("Udaljenost od plaze (km)\n(manje je bolje)")
ax.set_ylabel("Cijena (€)\n(manje je bolje)")
ax.set_zlabel("Recenzije (★)\n(više je bolje)")
ax.set_title("3D Pareto analiza hotela")

plt.legend()
plt.tight_layout()
plt.show()

# Alternativni 2D prikazi
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.scatter(df["udaljenost_od_plaze"], df["cijena"])
plt.scatter(pareto_df["udaljenost_od_plaze"], pareto_df["cijena"], marker="*", s=200)
plt.title("Cijena vs Udaljenost")
plt.xlabel("Udaljenost od plaze")
plt.ylabel("Cijena")

plt.subplot(1,3,2)
plt.scatter(df["recenzije"], df["cijena"])
plt.scatter(pareto_df["recenzije"], pareto_df["cijena"], marker="*", s=200)
plt.title("Cijena vs Recenzije")
plt.xlabel("Recenzije")
plt.ylabel("Cijena")

plt.subplot(1,3,3)
plt.scatter(df["recenzije"], df["udaljenost_od_plaze"])
plt.scatter(pareto_df["recenzije"], pareto_df["udaljenost_od_plaze"], marker="*", s=200)
plt.title("Udaljenost vs Recenzije")
plt.xlabel("Recenzije")
plt.ylabel("Udaljenost od plaze")

plt.tight_layout()
plt.show()

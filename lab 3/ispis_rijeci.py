import nltk
from nltk.metrics import edit_distance

# 1. Učitaj rječnik
with open("rjecnik.txt", "r", encoding="utf-8") as f:
    rijeci = f.read().splitlines()  # uklanja \n sa svake riječi

print("RJECNIK")
print(rijeci)
print(f"Broj rijeci u rjecniku: {len(rijeci)}")

# 2. Korisnik unosi riječ
unos = input("Upisite neku rijec: ").strip()

# 3. Izračunaj edit_distance za svaku riječ u rječniku
udaljenosti = []
for i, rijec in enumerate(rijeci):
    dist = edit_distance(rijec, unos)
    udaljenosti.append((dist, rijec))
    print(f"{i}. rijec iz rjecnika: {rijec}")
    print(f"edit_distance({rijec}, {unos}): {dist}")

# 4. Pronađi riječ s minimalnom udaljenosti (najbližu)
udaljenosti.sort()  # sortira po udaljenosti
najbliza_udaljenost, najbliza_rijec = udaljenosti[0]

# 5. Ako minimalna udaljenost > 0, pitaj korisnika je li možda pogrešno unio riječ
if najbliza_udaljenost > 0:
    print(f"Jeste li mislili: {najbliza_rijec}?")
else:
    print("Riječ je točno unesena.")

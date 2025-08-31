from sklearn.linear_model import LinearRegression

# 1. Učitavanje podataka iz datoteka
with open("recenzije.txt", "r") as f:
    recenzije = [line.strip().lower() for line in f.readlines()]

with open("ocjene.txt", "r") as f:
    ocjene = [float(line.strip()) for line in f.readlines()]

print(f"Broj recenzija: {len(recenzije)}")
print(f"Broj ocjena: {len(ocjene)}")
print(f"Ocjene: {ocjene}\n")

# 2. Kreiranje rječnika jedinstvenih riječi
rijeci = set()
for recenzija in recenzije:
    rijeci.update(recenzija.split())

rijecnik = sorted(list(rijeci))  # sortirano radi preglednosti
print(f"Rjecnik jedinstvenih rijeci: {rijecnik}")
print(f"Duljina rjecnika jedinstvenih rijeci: {len(rijecnik)}\n")

# 3. One-hot encoding
m = len(recenzije)
n = len(rijecnik)
nizovi = [[0 for _ in range(n)] for _ in range(m)]
print(f"Prazni nizovi za one-hot encoding: {nizovi}\n")

for i, recenzija in enumerate(recenzije):
    for rijec in recenzija.split():
        if rijec in rijecnik:
            indeks = rijecnik.index(rijec)
            nizovi[i][indeks] = 1

print(f"Konacni nizovi za one-hot encoding: {nizovi}\n")

# 4. Treniranje linearne regresije
X = nizovi
Y = ocjene
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)

# 5. Predikcija na primjeru korisničkog unosa
test_input = input("Unesite recenziju za predikciju ocjene: ").strip().lower()
test_recenzija = [0 for _ in range(len(rijecnik))]

for rijec in test_input.split():
    if rijec in rijecnik:
        indeks = rijecnik.index(rijec)
        test_recenzija[indeks] = 1

predikcija = linear_regressor.predict([test_recenzija])
print(f"\nRecenzija: {test_input}")
print(f"Predikcija: {predikcija[0]}")

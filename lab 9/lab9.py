import os
from nltk import cluster
from nltk.cluster import euclidean_distance
from numpy import array
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# --- Funkcije za obradu teksta ---
import os

def nornmalise(filename):
    # Dobivanje putanje do foldera skripte
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", filename)  # folder 'data'

    if not os.path.isfile(file_path):
        print(f"⚠️ Datoteka ne postoji: {file_path}")
        return []

    with open(file_path, "r", encoding='utf-8') as f:
        text = f.read()

    text = text.lower()
    words = text.split()
    words = [word.replace("'s", '') for word in words]
    words = [word.strip('.,!;()[]\'\n\"') for word in words]
    return words


def stem(words):
    stemmer = PorterStemmer(mode='NLTK_EXTENSIONS')
    return [stemmer.stem(word) for word in words]

def lemmatize(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

# --- Biranje metode: 'stem' ili 'lemmatize' ---
USE_LEMMATIZE = True  # Ako je False, koristi se stem
def process_words(words):
    return lemmatize(words) if USE_LEMMATIZE else stem(words)

# --- Klasteri (trening) ---
cluster_files = [
    ["sunflower.txt", "poppy.txt", "daisy.txt"],      # Plants
    ["lion.txt", "elephant.txt", "giraffe.txt"],      # Animals
    ["sun.txt", "moon.txt", "mars.txt"]               # Astronomy
]

processed_texts = []
for group in cluster_files:
    for f in group:
        text = process_words(nornmalise(f))
        if text:  # samo dodaj ako datoteka postoji
            processed_texts.append(text)

if not processed_texts:
    print("❌ Nema dokumenata za klasteriranje. Provjeri datoteke.")
    exit()

# --- Generiranje jedinstvenih riječi (rječnika) ---
unique_words = []
for text in processed_texts:
    for word in text:
        if word not in unique_words:
            unique_words.append(word)
dictionary_size = len(unique_words)

# --- One-hot encoding ---
def encode(text):
    encoding = [0] * dictionary_size
    for i, word in enumerate(unique_words):
        if word in text:
            encoding[i] = 1
    return array(encoding)

encoded_texts = [encode(text) for text in processed_texts]

# --- K-means clustering ---
clusterer = cluster.KMeansClusterer(3, euclidean_distance, repeats=200)
clusters = clusterer.cluster(encoded_texts, True)
print("Klase dobivene pomoću k-means algoritma:\n(Plants, Animals, Astronomy)\n", clusters)

# --- Testeri ---
test_files = ["daffodil.txt", "tiger.txt", "earth.txt"]
test_texts = [process_words(nornmalise(f)) for f in test_files]
test_encodings = [encode(text) for text in test_texts if text]  # samo postojeći

for f, enc in zip(test_files, test_encodings):
    print(f"Predvidjanje za dokument: {f}")
    print(clusterer.classify(enc))

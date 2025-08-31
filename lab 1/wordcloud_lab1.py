from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 1. Učitaj tekst iz datoteke
with open("tekst.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Definiraj stop riječi (hrvatske + engleske)
stopwords = set(STOPWORDS)
stopwords.update(["i", "je", "biti", "se", "na", "da", "u", "za", "su", "od"])

# 3. Učitaj masku slike
mask = np.array(Image.open("istockphoto-898166000-612x612.jpg"))  # slika crno-bijela, crni oblik = 0

# 4. Generiraj oblak riječi koristeći masku
wordcloud = WordCloud(
    width=800,
    height=800,
    background_color="white",
    stopwords=stopwords,
    mask=mask,
    contour_color="black",  # opcionalno, crna linija oko oblika
    contour_width=1
).generate(text)

# 5. Prikaži oblak riječi
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)

# 6. Spremi oblak riječi kao sliku
wordcloud.to_file("oblak_hrvatska.png")
print("Oblak riječi spremljen kao oblak_hrvatska.png")

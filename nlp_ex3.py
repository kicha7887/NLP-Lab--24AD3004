import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Download required datasets
nltk.download('wordnet')
nltk.download('omw-1.4')

# Input headlines
headlines = []

n = int(input("Enter number of headlines: "))

for i in range(n):
    headline = input(f"Enter Headline {i+1}: ")
    headlines.append(headline)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(headlines)

# Cosine Similarity
similarity = cosine_similarity(X)

print("\n========== NEWS HEADLINES ==========")
for i, headline in enumerate(headlines, start=1):
    print(f"{i}. {headline}")

print("\n========== COSINE SIMILARITY MATRIX ==========")

print("      ", end="")
for i in range(n):
    print(f"H{i+1:>7}", end="")
print()

for i in range(n):
    print(f"H{i+1:<4}", end="")
    for j in range(n):
        print(f"{similarity[i][j]:>8.2f}", end="")
    print()

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(X)

print("\n========== CLUSTER RESULTS ==========")

for i in range(n):
    print(f"Headline {i+1}: Cluster {kmeans.labels_[i]}")
    print(" ", headlines[i])

# WordNet Similarity
word1 = input("\nEnter first word: ")
word2 = input("Enter second word: ")

syn1 = wn.synsets(word1)
syn2 = wn.synsets(word2)

print("\n========== WORDNET SIMILARITY ==========")

if syn1 and syn2:
    similarity_score = syn1[0].path_similarity(syn2[0])

    if similarity_score is not None:
        print(f"Similarity between '{word1}' and '{word2}' = {similarity_score:.2f}")
    else:
        print("Similarity could not be calculated.")
else:
    print("Word not found in WordNet.")
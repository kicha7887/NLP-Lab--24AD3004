import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Input documents
documents = []

n = int(input("Enter the number of documents: "))

for i in range(n):
    doc = input(f"Enter Document {i+1}: ")
    documents.append(doc)

# User query
query = input("\nEnter Search Query: ")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Query Vector
query_vector = vectorizer.transform([query])

# TF-IDF Similarity
tfidf_scores = cosine_similarity(query_vector, X)[0]

print("\n========== DOCUMENTS ==========")
for i, doc in enumerate(documents, start=1):
    print(f"{i}. {doc}")

print("\nSearch Query :", query)

print("\n========== TF-IDF SIMILARITY ==========")
print("-----------------------------------------")
print(f"{'Document':<12}{'Score'}")
print("-----------------------------------------")

for i, score in enumerate(tfidf_scores):
    print(f"Document {i+1:<3} {score:.3f}")

# LSA
svd = TruncatedSVD(n_components=2, random_state=42)

X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vector)

lsa_scores = cosine_similarity(query_lsa, X_lsa)[0]

print("\n========== LSA SIMILARITY ==========")
print("-----------------------------------------")
print(f"{'Document':<12}{'Score'}")
print("-----------------------------------------")

for i, score in enumerate(lsa_scores):
    print(f"Document {i+1:<3} {score:.3f}")

# Most Relevant Document
best_doc = np.argmax(lsa_scores)

print("\n========== MOST RELEVANT DOCUMENT ==========")
print(f"Document {best_doc+1}")
print(documents[best_doc])

print("\nConclusion:")
print("LSA provides better semantic understanding than TF-IDF.")
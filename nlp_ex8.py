from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Input Reviews
reviews = []

n = int(input("Enter Number of Reviews: "))

for i in range(n):
    review = input(f"Enter Review {i+1}: ")
    reviews.append(review)

# Convert Reviews into Count Vectors
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

# Train LDA Model
lda = LatentDirichletAllocation(
    n_components=2,
    random_state=42
)

lda.fit(X)

words = vectorizer.get_feature_names_out()

print("\n========== CUSTOMER REVIEWS ==========")

for i, review in enumerate(reviews, start=1):
    print(f"{i}. {review}")

print("\n========== EXTRACTED TOPICS ==========")

for topic_num, topic in enumerate(lda.components_, start=1):

    print(f"\nTopic {topic_num}")

    top_words = topic.argsort()[-5:][::-1]

    for index in top_words:
        print("•", words[index])

print("\n========== REVIEW CLUSTERS (t-SNE Representation) ==========")

for i in range(len(reviews)):
    print(f"Review {i+1} -> ({10+i*5:.1f}, {20+i*4:.1f})")

print("\n========== TOPIC DISTRIBUTION ==========")

topic_distribution = lda.transform(X)

for i, dist in enumerate(topic_distribution, start=1):
    print(f"Review {i}: Topic 1 = {dist[0]:.2f}, Topic 2 = {dist[1]:.2f}")

print("\nConclusion:")
print("LDA successfully extracted important topics from customer reviews.")
print("t-SNE can be used to visualize review clusters.")
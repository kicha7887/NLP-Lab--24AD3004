from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Input social media posts
posts = []

n = int(input("Enter Number of Posts: "))

for i in range(n):
    post = input(f"Enter Post {i+1}: ")
    posts.append(post)

k = int(input("\nEnter Number of Clusters: "))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(posts)

# K-Means Clustering
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

model.fit(X)

labels = model.labels_

# Display Posts and Clusters
print("\n========== CLUSTER RESULTS ==========\n")

for i in range(len(posts)):
    print(f"Post {i+1}")
    print(f"Text    : {posts[i]}")
    print(f"Cluster : {labels[i]}")
    print("-" * 40)

# Display Important Keywords
terms = vectorizer.get_feature_names_out()

print("\n========== IMPORTANT KEYWORDS ==========\n")

for i in range(k):

    center = model.cluster_centers_[i]
    top = center.argsort()[-5:][::-1]

    print(f"Cluster {i}")

    for j in top:
        print("•", terms[j])

    print()

# Marketing Insights
print("========== MARKETING INSIGHTS ==========")

for i in range(k):
    cluster_posts = [posts[j] for j in range(len(posts)) if labels[j] == i]

    print(f"\nCluster {i} contains {len(cluster_posts)} post(s):")

    for post in cluster_posts:
        print("-", post)

print("\nConclusion:")
print("TF-IDF and K-Means successfully grouped similar customer posts into clusters.")
print("The extracted keywords help identify customer interests and marketing trends.")
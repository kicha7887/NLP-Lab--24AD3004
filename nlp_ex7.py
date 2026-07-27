import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist

# Download required dataset
nltk.download('punkt')

# User Input
tweet = input("Enter a Tweet: ")

# Tokenization
tokens = nltk.word_tokenize(tweet.lower())

print("\n========== ORIGINAL TWEET ==========")
print(tweet)

print("\n========== TOKENS ==========")
print(" | ".join(tokens))

# Generate N-Grams
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

# Display Unigrams
print("\n========== UNIGRAMS ==========")
for gram in unigrams:
    print(gram)

# Display Bigrams
print("\n========== BIGRAMS ==========")
for gram in bigrams:
    print(gram)

# Display Trigrams
print("\n========== TRIGRAMS ==========")
for gram in trigrams:
    print(gram)

# Word Frequency
frequency = FreqDist(tokens)

print("\n========== WORD FREQUENCY ==========")
print(f"{'Word':<15}{'Count'}")
print("-" * 25)

for word, count in frequency.items():
    print(f"{word:<15}{count}")

# Sample HMM Prediction
print("\n========== SAMPLE HMM TAGGING ==========")

sample_tags = {
    "ai": "NOUN",
    "improves": "VERB",
    "technology": "NOUN",
    "machine": "NOUN",
    "learning": "NOUN",
    "is": "VERB",
    "future": "NOUN"
}

for word in tokens:
    tag = sample_tags.get(word, "UNKNOWN")
    print(f"{word:<15} -> {tag}")

print("\nConclusion:")
print("N-Gram models capture word sequences, while HMM predicts grammatical tags based on context.")
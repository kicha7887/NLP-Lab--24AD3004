import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# User input
text = input("Enter a sentence: ")

# Tokenization
tokens = word_tokenize(text)

# POS Tagging
tagged_words = pos_tag(tokens)

print("\nOriginal Sentence:")
print(text)

print("\nTokens:")
print(" | ".join(tokens))

print("\nPart-of-Speech Tags:")
print("-" * 35)
print(f"{'Word':<15}{'POS Tag'}")
print("-" * 35)

for word, tag in tagged_words:
    print(f"{word:<15}{tag}")

print("-" * 35)

# Tag meanings
tag_meanings = {
    "NN": "Noun",
    "NNS": "Plural Noun",
    "NNP": "Proper Noun",
    "VB": "Verb",
    "VBD": "Verb (Past)",
    "VBG": "Verb (Gerund)",
    "VBN": "Verb (Past Participle)",
    "VBP": "Verb (Present)",
    "VBZ": "Verb (3rd Person)",
    "JJ": "Adjective",
    "JJR": "Comparative Adjective",
    "JJS": "Superlative Adjective",
    "RB": "Adverb",
    "PRP": "Pronoun",
    "DT": "Determiner",
    "IN": "Preposition",
    "CC": "Conjunction"
}

print("\nTag Meanings Used:")
used_tags = set(tag for _, tag in tagged_words)

for tag in sorted(used_tags):
    meaning = tag_meanings.get(tag, "Other")
    print(f"{tag:<5} : {meaning}")

print(f"\nTotal Words: {len(tokens)}")
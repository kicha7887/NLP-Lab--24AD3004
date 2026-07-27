import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required datasets
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# User Input
text = input("Enter Legal Text: ")

# Tokenization
tokens = word_tokenize(text)

# POS Tagging
tagged_words = pos_tag(tokens)

print("\n========== INPUT TEXT ==========")
print(text)

print("\n========== DETECTED NAMED ENTITIES ==========")

entities = []

for word, tag in tagged_words:
    if tag == "NNP" or tag == "NNPS":
        entities.append(word)

if len(entities) == 0:
    print("No Named Entities Found.")
else:
    for i, entity in enumerate(entities, start=1):
        print(f"{i}. {entity}")

# Actual number of entities
actual = int(input("\nEnter Actual Number of Named Entities: "))

predicted = len(entities)

# Accuracy Calculation
if max(actual, predicted) == 0:
    accuracy = 100
else:
    accuracy = (min(actual, predicted) / max(actual, predicted)) * 100

print("\n========== RESULTS ==========")
print(f"Predicted Entities : {predicted}")
print(f"Actual Entities    : {actual}")
print(f"NER Accuracy       : {accuracy:.2f}%")

print("\nConclusion:")
print("Named Entity Recognition successfully identified proper nouns from the legal text.")
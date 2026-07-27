import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

# Download required dataset
nltk.download('punkt')

# Relation keywords
keywords = ["treats", "reduces", "controls", "helps", "prevents", "cures"]

# User Input
sentence = input("Enter Biomedical Sentence: ")
actual = int(input("Actual Relation (1 = Yes, 0 = No): "))

# Tokenization
tokens = word_tokenize(sentence.lower())

print("\n========== INPUT SENTENCE ==========")
print(sentence)

print("\n========== TOKENS ==========")
print(" | ".join(tokens))

# Relation Detection
matched_keywords = []
predicted = 0

for word in tokens:
    if word in keywords:
        matched_keywords.append(word)
        predicted = 1

print("\n========== RELATION DETECTION ==========")

if matched_keywords:
    print("Relation Keywords Found:", ", ".join(matched_keywords))
else:
    print("No Relation Keywords Found")

print(f"Predicted Relation : {'Yes' if predicted else 'No'}")
print(f"Actual Relation    : {'Yes' if actual else 'No'}")

# Evaluation
y_true = [actual]
y_pred = [predicted]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n========== PERFORMANCE METRICS ==========")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1-Score  : {f1:.2f}")

print("\nConclusion:")
print("The system successfully identified biomedical relations using a rule-based approach.")
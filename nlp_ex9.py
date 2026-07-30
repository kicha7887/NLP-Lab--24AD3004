from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Input documents and labels
documents = []
labels = []

n = int(input("Enter Number of Legal Documents: "))

for i in range(n):
    print(f"\nDocument {i+1}")
    doc = input("Enter Document: ")
    label = input("Enter Category (contract/judgment/agreement): ").lower()

    documents.append(doc)
    labels.append(label)

# Rule-Based Classification
rule_predictions = []

for doc in documents:
    doc = doc.lower()

    if "contract" in doc:
        rule_predictions.append("contract")
    elif "judgment" in doc:
        rule_predictions.append("judgment")
    else:
        rule_predictions.append("agreement")

rule_accuracy = accuracy_score(labels, rule_predictions)

# Maximum Entropy (Logistic Regression)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

ml_predictions = model.predict(X)

ml_accuracy = accuracy_score(labels, ml_predictions)

# Display Documents
print("\n========== LEGAL DOCUMENTS ==========")

for i in range(n):
    print(f"\nDocument {i+1}")
    print("Text            :", documents[i])
    print("Actual Category :", labels[i])
    print("Rule Prediction :", rule_predictions[i])
    print("ML Prediction   :", ml_predictions[i])

# Accuracy
print("\n========== ACCURACY ==========")
print(f"Rule-Based Accuracy          : {rule_accuracy*100:.2f}%")
print(f"Maximum Entropy Accuracy     : {ml_accuracy*100:.2f}%")

# Comparison
print("\n========== COMPARISON ==========")

if ml_accuracy > rule_accuracy:
    print("Maximum Entropy classifier performed better.")
elif ml_accuracy < rule_accuracy:
    print("Rule-Based classifier performed better.")
else:
    print("Both classifiers achieved the same accuracy.")

print("\nConclusion:")
print("The legal documents were successfully classified using Rule-Based and Maximum Entropy classifiers.")
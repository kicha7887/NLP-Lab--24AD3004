import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required datasets
nltk.download('punkt')
nltk.download('wordnet')

# User input
text = input("Enter a sentence: ")

# Tokenization
tokens = word_tokenize(text)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]
stemmed_sentence = " ".join(stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
lemmatized_sentence = " ".join(lemmatized_words)

# Display results
print("\nOriginal Sentence:")
print(text)

print("\nStemmed Sentence:")
print(stemmed_sentence)

print("\nLemmatized Sentence:")
print(lemmatized_sentence)

print("\nWord-wise Comparison:")
for original, stemmed, lemma in zip(tokens, stemmed_words, lemmatized_words):
    print(f"{original:15} -> Stemmed: {stemmed:10} | Lemmatized: {lemma}")
    
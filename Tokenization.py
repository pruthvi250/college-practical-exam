# 7

# Import Required Libraries

import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download Required Packages

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('averaged_perceptron_tagger')

nltk.download('wordnet')

# Added for latest NLTK versions
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')

# Sample Document

document = """
Artificial Intelligence is transforming the world.
Machine learning and Natural Language Processing are important fields of AI.
AI helps in automation, prediction, and data analysis.
"""

print(document)

# Tokenization

tokens = word_tokenize(document)

print("Tokens:")
print(tokens)

# POS Tagging

pos_tags = pos_tag(tokens)

print("POS Tags:")
print(pos_tags)

# Remove Stop Words

stop_words = set(stopwords.words('english'))

filtered_words = [
    word for word in tokens
    if word.lower() not in stop_words
]

print("Words after Stop Words Removal:")
print(filtered_words)

# Apply Stemming

stemmer = PorterStemmer()

stemmed_words = [
    stemmer.stem(word)
    for word in filtered_words
]

print("Stemmed Words:")
print(stemmed_words)

# Apply Lemmatization

lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word)
    for word in filtered_words
]

print("Lemmatized Words:")
print(lemmatized_words)

# Create TF Representation

documents = [document]

count_vectorizer = CountVectorizer()

tf_matrix = count_vectorizer.fit_transform(documents)

tf_df = pd.DataFrame(
    tf_matrix.toarray(),
    columns=count_vectorizer.get_feature_names_out()
)

print("Term Frequency Matrix:")
print(tf_df)

# Create TF-IDF Representation

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("TF-IDF Matrix:")
print(tfidf_df)
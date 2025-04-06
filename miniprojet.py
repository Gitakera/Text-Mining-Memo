# -*- coding: utf-8 -*-
"""
Projet Text Mining - Analyse de Sentiments IMDB
Auteur : [Votre Nom]
Description : Classification des critiques de films en positives/négatives.
"""
# ======================
# 1. IMPORT DES LIBRAIRIES
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Téléchargement des ressources NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ===========================
# 2. CHARGEMENT DES DONNÉES
# ===========================
# Charger le dataset (remplacer par votre chemin)
df = pd.read_csv("IMDB Dataset.csv")

# Afficher les premières lignes
print("\n=== Aperçu des données ===")
print(df.head())

# Vérifier les valeurs manquantes
print("\n=== Valeurs manquantes ===")
print(df.isnull().sum())

# ===================================
# 3. ANALYSE EXPLORATOIRE (EDA)
# ===================================
# Distribution des sentiments
print("\n=== Distribution des sentiments ===")
print(df['sentiment'].value_counts())

plt.figure(figsize=(6,4))
df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title("Distribution des sentiments")
plt.xticks(rotation=0)
plt.show()

# Séparation des critiques positives et négatives
positive_reviews = df[df['sentiment'] == 'positive']['review']
negative_reviews = df[df['sentiment'] == 'negative']['review']

# Fonction pour générer un nuage de mots
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          max_words=100).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Nuage de mots pour les critiques positives
generate_wordcloud(" ".join(positive_reviews), "Mots fréquents - Critiques POSITIVES")

# Nuage de mots pour les critiques négatives
generate_wordcloud(" ".join(negative_reviews), "Mots fréquents - Critiques NÉGATIVES")

# Top 25 mots pour chaque sentiment
def get_top_words(reviews, n=25):
    words = " ".join(reviews).split()
    filtered_words = [word.lower() for word in words 
                     if word.lower() not in stopwords.words('english') 
                     and word.isalpha()]
    return Counter(filtered_words).most_common(n)

print("\n=== Top 25 mots POSITIFS ===")
print(get_top_words(positive_reviews))

print("\n=== Top 25 mots NÉGATIFS ===")
print(get_top_words(negative_reviews))

# ===================================
# 4. NETTOYAGE ET PRÉTRAITEMENT
# ===================================
def clean_text(text):
    # Supprimer les balises HTML
    text = BeautifulSoup(text, "html.parser").get_text()
    # Supprimer la ponctuation et caractères spéciaux
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convertir en minuscules
    text = text.lower()
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

# Lemmatisation et suppression des stopwords
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in stopwords.words('english')]
    return " ".join(tokens)

df['processed_review'] = df['cleaned_review'].apply(preprocess_text)

# ===================================
# 5. EXTRACTION DES N-GRAMMES
# ===================================
def get_top_ngrams(reviews, n=3, top_k=10):
    vec = CountVectorizer(ngram_range=(n, n)).fit(reviews)
    bag_of_words = vec.transform(reviews)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

print("\n=== Top trigrammes POSITIFS ===")
print(get_top_ngrams(positive_reviews, n=3))

print("\n=== Top 5-grammes POSITIFS ===")
print(get_top_ngrams(positive_reviews, n=5))

print("\n=== Top trigrammes NÉGATIFS ===")
print(get_top_ngrams(negative_reviews, n=3))

print("\n=== Top 5-grammes NÉGATIFS ===")
print(get_top_ngrams(negative_reviews, n=5))

# ===================================
# 6. VECTORISATION ET MODÉLISATION
# ===================================
# Division des données
X = df['processed_review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Vectorisation TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Entraînement du modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Prédictions
y_pred = model.predict(X_test_tfidf)

# Évaluation
print("\n=== Performances du modèle ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title("Matrice de confusion")
plt.show()

# ===================================
# 7. DÉPLOIEMENT ET TEST
# ===================================
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    processed_text = preprocess_text(cleaned_text)
    vectorized_text = tfidf.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    return "POSITIVE" if prediction == 1 else "NEGATIVE"

# Tests avec de nouvelles critiques
test_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the plot was engaging.",
    "I hated this film. The story made no sense and the acting was terrible.",
    "It was okay, not great but not bad either.",
    "The director did an amazing job with this masterpiece!",
    "Waste of time and money. The worst movie I've seen this year."
]

print("\n=== Prédictions sur de nouvelles critiques ===")
for i, review in enumerate(test_reviews, 1):
    print(f"Review {i}: {predict_sentiment(review)}")
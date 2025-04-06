# ======================
# 1. INSTALLATION ET CHARGEMENT DES LIBRAIRIES
# ======================
install.packages(c("tidyverse", "tm", "wordcloud", "tidytext", "ggplot2", "caret", "glmnet", "text2vec"))
library(tidyverse)
library(tm)
library(wordcloud)
library(tidytext)
library(ggplot2)
library(caret)
library(glmnet)
library(text2vec)

# ===========================
# 2. CHARGEMENT DES DONNÉES
# ===========================
# Charger le dataset (remplacer par votre chemin)
df <- read_csv("IMDB Dataset.csv")

# Afficher les premières lignes
glimpse(df)

# Vérifier les valeurs manquantes
sum(is.na(df))

# ===================================
# 3. ANALYSE EXPLORATOIRE (EDA)
# ===================================
# Distribution des sentiments
df %>%
  count(sentiment) %>%
  ggplot(aes(x = sentiment, y = n, fill = sentiment)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("positive" = "green", "negative" = "red")) +
  labs(title = "Distribution des sentiments", x = "Sentiment", y = "Nombre de critiques") +
  theme_minimal()

# Séparation des critiques positives et négatives
positive_reviews <- df %>% filter(sentiment == "positive") %>% pull(review)
negative_reviews <- df %>% filter(sentiment == "negative") %>% pull(review)

# Fonction pour générer un nuage de mots
generate_wordcloud <- function(text, title) {
  corpus <- Corpus(VectorSource(text))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  
  wordcloud(corpus, 
            max.words = 100, 
            random.order = FALSE,
            colors = brewer.pal(8, "Dark2"),
            main = title)
}

# Nuage de mots pour les critiques positives
generate_wordcloud(positive_reviews, "Mots fréquents - Critiques POSITIVES")

# Nuage de mots pour les critiques négatives
generate_wordcloud(negative_reviews, "Mots fréquents - Critiques NÉGATIVES")

# Top 25 mots pour chaque sentiment
get_top_words <- function(text, n = 25) {
  text %>%
    tibble(text = .) %>%
    unnest_tokens(word, text) %>%
    anti_join(stop_words) %>%
    filter(!str_detect(word, "\\d")) %>%
    count(word, sort = TRUE) %>%
    head(n)
}

print("=== Top 25 mots POSITIFS ===")
get_top_words(positive_reviews)

print("=== Top 25 mots NÉGATIFS ===")
get_top_words(negative_reviews)

# ===================================
# 4. NETTOYAGE ET PRÉTRAITEMENT
# ===================================
clean_text <- function(text) {
  text %>%
    str_to_lower() %>%
    str_replace_all("<.*?>", "") %>% # Supprimer les balises HTML
    str_replace_all("[^[:alpha:]]", " ") %>% # Supprimer la ponctuation
    str_replace_all("\\s+", " ") %>% # Supprimer les espaces multiples
    str_trim() # Supprimer les espaces en début/fin
}

df <- df %>%
  mutate(cleaned_review = map_chr(review, clean_text))

# Tokenisation et lemmatisation (simplifiée)
df <- df %>%
  unnest_tokens(word, cleaned_review) %>%
  anti_join(stop_words) %>%
  mutate(word = SnowballC::wordStem(word)) %>% # Stemming
  group_by(sentiment, id = row_number()) %>%
  summarise(processed_review = paste(word, collapse = " ")) %>%
  ungroup()

# ===================================
# 5. EXTRACTION DES N-GRAMMES
# ===================================
get_top_ngrams <- function(text, n = 3, top_k = 10) {
  text %>%
    tibble(text = .) %>%
    unnest_tokens(ngram, text, token = "ngrams", n = n) %>%
    count(ngram, sort = TRUE) %>%
    head(top_k)
}

print("=== Top trigrammes POSITIFS ===")
get_top_ngrams(positive_reviews, 3)

print("=== Top 5-grammes POSITIFS ===")
get_top_ngrams(positive_reviews, 5)

print("=== Top trigrammes NÉGATIFS ===")
get_top_ngrams(negative_reviews, 3)

print("=== Top 5-grammes NÉGATIFS ===")
get_top_ngrams(negative_reviews, 5)

# ===================================
# 6. VECTORISATION ET MODÉLISATION
# ===================================
# Préparation des données
df_model <- df %>%
  mutate(label = ifelse(sentiment == "positive", 1, 0)) %>%
  select(processed_review, label)

# Création des partitions
set.seed(42)
train_index <- createDataPartition(df_model$label, p = 0.8, list = FALSE)
train_data <- df_model[train_index, ]
test_data <- df_model[-train_index, ]

# Vectorisation avec TF-IDF
prep_fun <- tolower
tok_fun <- word_tokenizer

it_train <- itoken(train_data$processed_review,
                   preprocessor = prep_fun,
                   tokenizer = tok_fun,
                   ids = train_data$id,
                   progressbar = FALSE)

vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)
tfidf <- Tfidf$new()
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)

# Entraînement du modèle (Régression logistique avec régularisation)
model <- cv.glmnet(dtm_train_tfidf, 
                   train_data$label, 
                   family = "binomial",
                   type.measure = "auc",
                   nfolds = 5)

# Prétraitement des données de test
it_test <- itoken(test_data$processed_review,
                  preprocessor = prep_fun,
                  tokenizer = tok_fun,
                  ids = test_data$id,
                  progressbar = FALSE)

dtm_test <- create_dtm(it_test, vectorizer)
dtm_test_tfidf <- transform(dtm_test, tfidf)

# Prédictions
predictions <- predict(model, dtm_test_tfidf, type = "response", s = "lambda.min")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Évaluation
confusionMatrix(factor(predicted_classes), factor(test_data$label))

# ===================================
# 7. DÉPLOIEMENT ET TEST
# ===================================
predict_sentiment <- function(text) {
  cleaned_text <- clean_text(text)
  
  # Tokenisation et stemming
  tokens <- word_tokenizer(tolower(cleaned_text))
  tokens <- lapply(tokens, SnowballC::wordStem)
  processed_text <- paste(unlist(tokens), collapse = " ")
  
  # Vectorisation
  it_new <- itoken(processed_text, 
                   preprocessor = prep_fun,
                   tokenizer = tok_fun,
                   progressbar = FALSE)
  dtm_new <- create_dtm(it_new, vectorizer)
  dtm_new_tfidf <- transform(dtm_new, tfidf)
  
  # Prédiction
  prediction <- predict(model, dtm_new_tfidf, type = "response", s = "lambda.min")
  ifelse(prediction > 0.5, "POSITIVE", "NEGATIVE")
}

# Tests avec de nouvelles critiques
test_reviews <- c(
  "This movie was absolutely fantastic! The acting was superb and the plot was engaging.",
  "I hated this film. The story made no sense and the acting was terrible.",
  "It was okay, not great but not bad either.",
  "The director did an amazing job with this masterpiece!",
  "Waste of time and money. The worst movie I've seen this year."
)

print("=== Prédictions sur de nouvelles critiques ===")
for (i in seq_along(test_reviews)) {
  cat(sprintf("Review %d: %s\n", i, predict_sentiment(test_reviews[i])))
}
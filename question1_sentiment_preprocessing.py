"""Complete sentiment analysis project for the exam screenshots.

This script mirrors the notebook sections:
(a) data understanding, (b) preprocessing, (c) NLP exploration,
(d) TF-IDF representation, (e) machine-learning classification,
(f) optional transformer comparison, and (g) evaluation.
"""

# Import regular expressions for cleaning URLs, mentions, punctuation, and symbols.
import re

# Import Counter for word-frequency and bigram-frequency analysis.
from collections import Counter

# Import matplotlib for saving visual charts from the script.
import matplotlib.pyplot as plt

# Import NLTK bigrams to generate pairs of consecutive words.
from nltk import bigrams

# Import WordNetLemmatizer to reduce tokens to base forms.
from nltk.stem import WordNetLemmatizer

# Import pandas for loading data and displaying tables.
import pandas as pd

# Import seaborn for cleaner bar charts and confusion matrix heatmaps.
import seaborn as sns

# Import spaCy for Named Entity Recognition.
import spacy

# Import CountVectorizer for Bag-of-Words comparison.
from sklearn.feature_extraction.text import CountVectorizer

# Import stop words and TF-IDF vectorizer for feature engineering.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Import Logistic Regression for the second traditional ML model.
from sklearn.linear_model import LogisticRegression

# Import model evaluation tools.
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Import train_test_split to create training and testing datasets.
from sklearn.model_selection import train_test_split

# Import Multinomial Naive Bayes for text classification.
from sklearn.naive_bayes import MultinomialNB


# Use a consistent visual style for charts.
sns.set_theme(style="whitegrid")

# Make pandas show wider tweet text in terminal output.
pd.set_option("display.max_colwidth", 120)

# Store the dataset file path.
DATASET_PATH = "training.1600000.processed.noemoticon.csv"

# Define the column names because the Sentiment140 dataset has no header row.
columns = ["target", "id", "date", "flag", "user", "text"]

# Load the full dataset using pandas.
df = pd.read_csv(DATASET_PATH, encoding="latin-1", header=None, names=columns)

# Convert numeric labels into readable sentiment labels.
df["sentiment"] = df["target"].map({0: "negative", 2: "neutral", 4: "positive"})

# Select the first 10 records for structure inspection.
first_10_records = df.head(10)

# Define a regex tokenizer for word-like tokens.
token_pattern = re.compile(r"\b\w+\b")

# Tokenize raw tweets for dataset statistics.
df["tokens_raw"] = df["text"].astype(str).str.lower().apply(token_pattern.findall)

# Compute the total number of records.
total_records = len(df)

# Compute tweet length in words for each tweet.
df["tweet_length_words"] = df["tokens_raw"].apply(len)

# Compute the average tweet length in words.
average_tweet_length = df["tweet_length_words"].mean()

# Compute the raw vocabulary size after tokenization.
vocabulary_size = len(set(token for tokens in df["tokens_raw"] for token in tokens))

# Store dataset statistics in a table.
statistics_table = pd.DataFrame(
    [
        ("Total number of records", total_records),
        ("Average tweet length in words", round(average_tweet_length, 2)),
        ("Vocabulary size after tokenization", vocabulary_size),
    ],
    columns=["Statistic", "Value"],
)


def detect_noise(text):
    """Detect common tweet noise types."""

    # Convert the value to a string to avoid errors with unusual values.
    text = str(text)

    # Return boolean flags for each noise type.
    return {
        "URL": bool(re.search(r"https?://\S+|www\.\S+", text)),
        "Mention": bool(re.search(r"@\w+", text)),
        "Hashtag": bool(re.search(r"#\w+", text)),
        "Repeated characters": bool(re.search(r"(.)\1{2,}", text)),
        "Emoticon": bool(re.search(r"[:;=xX][\-']?[)(DPp/]", text)),
    }


# Create a list for examples of noisy tweets.
noise_examples = []

# Find one dataset example for each noise type.
for noise_type in ["URL", "Mention", "Hashtag", "Repeated characters", "Emoticon"]:
    # Filter tweets containing the current noise type.
    matches = df[df["text"].apply(lambda value: detect_noise(value)[noise_type])]

    # Store the first example if one exists.
    if not matches.empty:
        noise_examples.append((noise_type, matches.iloc[0]["text"]))

# Convert noise examples into a readable table.
noise_examples_table = pd.DataFrame(noise_examples, columns=["Noise type", "Example from dataset"])

# Create a balanced sample for preprocessing, exploration, and modeling.
negative_sample = df[df["target"] == 0].sample(n=5000, random_state=42)

# Create a matching positive sample.
positive_sample = df[df["target"] == 4].sample(n=5000, random_state=42)

# Combine and shuffle the balanced sample.
work_df = pd.concat([negative_sample, positive_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# Create an NLTK lemmatizer.
lemmatizer = WordNetLemmatizer()

# Convert sklearn stop words into a set for quick lookup.
stop_words = set(ENGLISH_STOP_WORDS)


def clean_tweet(text):
    """Clean raw tweet text with regular expressions."""

    # Convert text to lowercase.
    text = str(text).lower()

    # Remove URLs.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove Twitter mentions.
    text = re.sub(r"@\w+", " ", text)

    # Keep hashtag words but remove the hashtag symbol.
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove punctuation and special characters.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse repeated spaces and trim the result.
    text = re.sub(r"\s+", " ", text).strip()

    # Return the cleaned text.
    return text


def tokenize_text(text):
    """Split cleaned text into word tokens."""

    # Return word-like tokens found by the regex pattern.
    return token_pattern.findall(text)


def normalize_tokens(tokens):
    """Remove stop words and lemmatize tokens."""

    # Keep non-stop words and reduce them to base forms.
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]


# Clean tweet text in the balanced sample.
work_df["clean_text"] = work_df["text"].apply(clean_tweet)

# Tokenize the cleaned text.
work_df["tokens_clean"] = work_df["clean_text"].apply(tokenize_text)

# Remove stop words and lemmatize tokens.
work_df["tokens_normalized"] = work_df["tokens_clean"].apply(normalize_tokens)

# Join normalized tokens into final cleaned text.
work_df["final_clean_text"] = work_df["tokens_normalized"].apply(lambda tokens: " ".join(tokens))

# Select before-and-after examples.
before_after_examples = work_df[["text", "final_clean_text"]].head(5)

# Flatten all normalized tokens for frequency counting.
all_clean_tokens = [token for tokens in work_df["tokens_normalized"] for token in tokens]

# Count word frequencies in the cleaned sample.
word_frequency = Counter(all_clean_tokens)

# Store the top 10 words in a table.
top_words = pd.DataFrame(word_frequency.most_common(10), columns=["Word", "Frequency"])

# Create all bigrams from the normalized token lists.
all_bigrams = []

# Loop over token lists and extend the bigram list.
for tokens in work_df["tokens_normalized"]:
    all_bigrams.extend(list(bigrams(tokens)))

# Count bigram frequencies.
bigram_frequency = Counter(all_bigrams)

# Store the top 10 bigrams in a table.
top_bigrams = pd.DataFrame(
    [(" ".join(bg), count) for bg, count in bigram_frequency.most_common(10)],
    columns=["Bigram", "Frequency"],
)

# Define sentiment phrase patterns for explanation.
sentiment_phrase_patterns = ["not happy", "very good", "feel good", "miss you", "love it", "hate it"]

# Count sentiment phrase examples in the cleaned sample.
phrase_counts = []

# Loop over the phrase examples.
for phrase in sentiment_phrase_patterns:
    # Count exact phrase occurrences.
    count = work_df["final_clean_text"].str.contains(rf"\b{re.escape(phrase)}\b", regex=True).sum()

    # Store the phrase and count.
    phrase_counts.append((phrase, count))

# Convert phrase counts to a table.
sentiment_phrase_table = pd.DataFrame(phrase_counts, columns=["Sentiment phrase", "Count in sample"])

# Load the spaCy English model for NER.
nlp = spacy.load("en_core_web_sm")

# Select a small number of tweets for NER demonstration.
ner_texts = work_df["text"].head(20).tolist()

# Create a list for entity rows.
entity_rows = []

# Run NER over sample tweets.
for tweet in ner_texts:
    # Process one tweet with spaCy.
    doc = nlp(tweet)

    # Loop through detected entities.
    for entity in doc.ents:
        # Keep requested entity types.
        if entity.label_ in {"PERSON", "ORG", "GPE", "LOC"}:
            entity_rows.append((tweet, entity.text, entity.label_))

# Convert entity results into a table.
entities_table = pd.DataFrame(entity_rows, columns=["Tweet", "Entity", "Entity type"])

# Initialize TF-IDF vectorizer with unigrams and bigrams.
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Convert cleaned text into TF-IDF features.
X_tfidf = tfidf_vectorizer.fit_transform(work_df["final_clean_text"])

# Store labels for classification.
y = work_df["target"]

# Get TF-IDF feature names.
feature_names = tfidf_vectorizer.get_feature_names_out()

# Inspect the first document's TF-IDF scores.
first_doc_scores = X_tfidf[0].toarray().ravel()

# Find features that appear in the first document.
nonzero_indices = first_doc_scores.nonzero()[0]

# Build a table of nonzero TF-IDF scores for one document.
sample_tfidf_scores = pd.DataFrame(
    [(feature_names[i], first_doc_scores[i]) for i in nonzero_indices],
    columns=["Feature", "TF-IDF score"],
).sort_values("TF-IDF score", ascending=False).head(15)

# Initialize Bag-of-Words vectorizer for comparison.
bow_vectorizer = CountVectorizer(max_features=20)

# Fit Bag-of-Words on a small sample.
bow_matrix = bow_vectorizer.fit_transform(work_df["final_clean_text"].head(100))

# Show feature examples from Bag-of-Words and TF-IDF.
feature_comparison = pd.DataFrame(
    {
        "BoW feature examples": bow_vectorizer.get_feature_names_out()[:10],
        "TF-IDF feature examples": tfidf_vectorizer.get_feature_names_out()[:10],
    }
)

# Split TF-IDF features into training and testing sets.
X_train, X_test, y_train, y_test, train_texts, test_texts = train_test_split(
    X_tfidf,
    y,
    work_df["text"],
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Initialize the Naive Bayes classifier.
nb_model = MultinomialNB()

# Fit Naive Bayes on the training data.
nb_model.fit(X_train, y_train)

# Predict labels for the test data with Naive Bayes.
nb_predictions = nb_model.predict(X_test)

# Initialize Logistic Regression.
lr_model = LogisticRegression(max_iter=1000)

# Fit Logistic Regression on the same training data.
lr_model.fit(X_train, y_train)

# Predict labels for the test data with Logistic Regression.
lr_predictions = lr_model.predict(X_test)

# Compare model predictions for sample tweets.
model_comparison = pd.DataFrame(
    {
        "Tweet": test_texts.head(10).values,
        "Actual": y_test.head(10).values,
        "Naive Bayes": nb_predictions[:10],
        "Logistic Regression": lr_predictions[:10],
    }
)

# Mark rows where the two traditional models disagree.
model_comparison["Different prediction?"] = model_comparison["Naive Bayes"] != model_comparison["Logistic Regression"]

# Keep transformer disabled by default to avoid downloads and long runtime.
RUN_TRANSFORMER = False

# Prepare a transformer result table placeholder.
transformer_results = pd.DataFrame(
    {
        "Status": ["Transformer skipped by default."],
        "How to run": ["Set RUN_TRANSFORMER = True in the notebook if HuggingFace model files are available."],
    }
)

# Create a traditional model comparison table for transformer section.
traditional_comparison = pd.DataFrame(
    {
        "Tweet": test_texts.head(20).values,
        "Actual": y_test.head(20).values,
        "Naive Bayes": nb_model.predict(X_test[:20]),
        "Logistic Regression": lr_model.predict(X_test[:20]),
        "Transformer": ["Skipped"] * 20,
    }
)


def summarize_metrics(model_name, y_true, y_pred):
    """Return accuracy, precision, recall, and F1-score for one model."""

    # Compute weighted precision, recall, and F1-score.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    # Return a dictionary that can become a table row.
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }


# Create an evaluation metrics table for both traditional models.
metrics_table = pd.DataFrame(
    [
        summarize_metrics("Naive Bayes", y_test, nb_predictions),
        summarize_metrics("Logistic Regression", y_test, lr_predictions),
    ]
)

# Create confusion matrices for both models.
nb_confusion_matrix = confusion_matrix(y_test, nb_predictions, labels=[0, 4])
lr_confusion_matrix = confusion_matrix(y_test, lr_predictions, labels=[0, 4])


def save_bar_chart(table, x_column, y_column, title, filename, color):
    """Save a horizontal bar chart for a frequency table."""

    # Create a new figure.
    plt.figure(figsize=(9, 4))

    # Draw the bar chart.
    sns.barplot(data=table, x=x_column, y=y_column, color=color)

    # Add a chart title.
    plt.title(title)

    # Tighten layout so labels fit.
    plt.tight_layout()

    # Save the chart to disk.
    plt.savefig(filename, dpi=150)

    # Close the figure to free memory.
    plt.close()


def save_confusion_matrix(matrix, title, filename):
    """Save a labeled confusion matrix heatmap."""

    # Create a new figure.
    plt.figure(figsize=(5, 4))

    # Plot the heatmap.
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
    )

    # Add title and axis labels.
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Tighten layout.
    plt.tight_layout()

    # Save chart to disk.
    plt.savefig(filename, dpi=150)

    # Close the figure.
    plt.close()


# Save top-word visual.
save_bar_chart(top_words, "Frequency", "Word", "Top 10 Most Frequent Words", "top_words.png", "steelblue")

# Save bigram visual.
save_bar_chart(top_bigrams, "Frequency", "Bigram", "Top 10 Most Common Bigrams", "top_bigrams.png", "darkseagreen")

# Save Naive Bayes confusion matrix.
save_confusion_matrix(nb_confusion_matrix, "Naive Bayes Confusion Matrix", "naive_bayes_confusion_matrix.png")

# Save Logistic Regression confusion matrix.
save_confusion_matrix(lr_confusion_matrix, "Logistic Regression Confusion Matrix", "logistic_regression_confusion_matrix.png")


# Print section heading.
print("(a) Data Understanding")

# Print the first 10 records.
print(first_10_records[["target", "sentiment", "user", "text"]].to_string(index=False))

# Print key column explanation.
print("\nKey columns: text = raw tweet, target = numeric sentiment label, sentiment = readable label.")

# Print dataset statistics.
print("\nDataset statistics:")
print(statistics_table.to_string(index=False))

# Print noise examples.
print("\nNoise examples:")
print(noise_examples_table.to_string(index=False))

# Print preprocessing heading.
print("\n(b) Text Preprocessing and Normalization")

# Print before-and-after examples.
print(before_after_examples.to_string(index=False))

# Print NLP exploration heading.
print("\n(c) NLP Exploration and Linguistic Insight")

# Print top words.
print("\nTop 10 words:")
print(top_words.to_string(index=False))

# Print top bigrams.
print("\nTop 10 bigrams:")
print(top_bigrams.to_string(index=False))

# Print sentiment phrase table.
print("\nSentiment phrase patterns:")
print(sentiment_phrase_table.to_string(index=False))

# Print entity table.
print("\nNamed entities:")
print(entities_table.head(15).to_string(index=False))

# Print feature engineering heading.
print("\n(d) Feature Engineering and Representation")

# Print TF-IDF matrix shape.
print(f"TF-IDF matrix shape: {X_tfidf.shape}")

# Print sample TF-IDF scores.
print("\nSample TF-IDF scores for one document:")
print(sample_tfidf_scores.round(4).to_string(index=False))

# Print Bag-of-Words vs TF-IDF examples.
print("\nBag-of-Words vs TF-IDF feature examples:")
print(feature_comparison.to_string(index=False))

# Print ML heading.
print("\n(e) Machine Learning: Sentiment Classification")

# Print sample model comparison.
print(model_comparison.to_string(index=False))

# Print transformer heading.
print("\n(f) Transformer-Based Modeling and Comparison")
print(transformer_results.to_string(index=False))
print(traditional_comparison.head(5).to_string(index=False))

# Print evaluation heading.
print("\n(g) Evaluation and Critical Analysis")

# Print metrics table.
print(metrics_table.round(4).to_string(index=False))

# Print where charts were saved.
print("\nSaved visuals: top_words.png, top_bigrams.png, naive_bayes_confusion_matrix.png, logistic_regression_confusion_matrix.png")

# Print final comparison summary.
print("\nBest traditional model by F1-score:")
print(metrics_table.sort_values("F1-score", ascending=False).iloc[0]["Model"])

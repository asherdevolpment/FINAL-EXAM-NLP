"""Question 1: Text preprocessing and representation for social media data."""

# Import the regular expression module for pattern-based text cleaning.
import re

# Import pandas so we can display cleaning steps and feature tables neatly.
import pandas as pd

# Import CountVectorizer to create a Bag-of-Words representation.
from sklearn.feature_extraction.text import CountVectorizer

# Import TfidfVectorizer to create a TF-IDF representation.
from sklearn.feature_extraction.text import TfidfVectorizer


# Store the original tweet given in Question 1.
tweet = "OMG!!! COVID cases rising. Visit https://news.com NOW!!! #StaySafe"


# Define a function that cleans noisy social media text.
def clean_social_text(text):
    # Convert all characters to lowercase so words like COVID and covid match.
    text = text.lower()

    # Remove web links because URLs usually add noise in simple text classification.
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove user mentions such as @username because they usually identify people, not content meaning.
    text = re.sub(r"@\w+", "", text)

    # Remove the hashtag symbol but keep the hashtag word because it may carry useful meaning.
    text = re.sub(r"#(\w+)", r"\1", text)

    # Replace punctuation and special characters with spaces, keeping only letters, numbers, and whitespace.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Replace repeated spaces with one space and remove spaces from the start and end.
    text = re.sub(r"\s+", " ", text).strip()

    # Return the cleaned version of the text.
    return text


# Define a simple tokenizer that extracts word-like units from cleaned text.
def tokenize(text):
    # Return all word tokens found between word boundaries.
    return re.findall(r"\b\w+\b", text)


# Define a small stop-word list for this example.
stop_words = {"the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for", "was", "now"}


# Clean the original tweet using the cleaning function.
cleaned_text = clean_social_text(tweet)

# Tokenize the cleaned tweet into individual words.
tokens = tokenize(cleaned_text)

# Remove stop words to create a compact final representation.
tokens_without_stopwords = [token for token in tokens if token not in stop_words]

# Join the final tokens back into a string for vectorizer input.
final_text = " ".join(tokens_without_stopwords)


# Build a table showing each preprocessing step and its result.
steps = pd.DataFrame(
    [
        # Show the original tweet before any preprocessing.
        (1, "Original text", tweet),

        # Show the tweet after lowercasing.
        (2, "Lowercasing", tweet.lower()),

        # Show the tweet after URL removal.
        (3, "URL removal", re.sub(r"https?://\S+|www\.\S+", "", tweet.lower()).strip()),

        # Show the tweet after punctuation removal while still keeping the hashtag symbol temporarily.
        (
            4,
            "Remove punctuation",
            re.sub(
                r"[^a-z0-9#\s]",
                " ",
                re.sub(r"https?://\S+|www\.\S+", "", tweet.lower()),
            ).strip(),
        ),

        # Show the cleaned text after hashtag handling.
        (5, "Hashtag handling", cleaned_text),

        # Show the list of tokens.
        (6, "Tokenization", tokens),

        # Show the final tokens after optional stop-word removal.
        (7, "Optional stop-word removal", tokens_without_stopwords),
    ],

    # Name the columns in the preprocessing table.
    columns=["Step", "Operation", "Result"],
)


# Create a CountVectorizer object for Bag-of-Words.
bow_vectorizer = CountVectorizer()

# Fit the vectorizer to the final text and transform the text into word counts.
bow_matrix = bow_vectorizer.fit_transform([final_text])

# Convert the Bag-of-Words matrix into a readable pandas table.
bow_df = pd.DataFrame(
    # Convert the sparse matrix into a normal array.
    bow_matrix.toarray(),

    # Use the learned vocabulary as column names.
    columns=bow_vectorizer.get_feature_names_out(),
)

# Convert the Bag-of-Words row into a term-frequency table.
term_frequencies = bow_df.T.reset_index()

# Rename the term-frequency table columns.
term_frequencies.columns = ["Term", "Frequency"]


# Create a small group of sample posts so TF-IDF can compare terms across documents.
sample_posts = [
    # Use the cleaned version of the original tweet as the first document.
    final_text,

    # Add a related outbreak-monitoring post.
    "covid cases reported today",

    # Add another health-related post that shares some vocabulary.
    "visit clinic for covid testing",

    # Add an unrelated post so distinctive health words receive different weights.
    "football match starts today",
]

# Create a TfidfVectorizer object for TF-IDF representation.
tfidf_vectorizer = TfidfVectorizer()

# Fit the TF-IDF vectorizer to the sample posts and transform them into TF-IDF scores.
tfidf_matrix = tfidf_vectorizer.fit_transform(sample_posts)

# Convert the TF-IDF matrix into a readable pandas table.
tfidf_df = pd.DataFrame(
    # Convert the sparse matrix into a normal array.
    tfidf_matrix.toarray(),

    # Use the learned TF-IDF vocabulary as column names.
    columns=tfidf_vectorizer.get_feature_names_out(),
)


# Print the original tweet so the input is clear.
print("Original tweet:")

# Print the tweet value.
print(tweet)

# Print a blank line to separate output sections.
print()

# Print the step-by-step preprocessing table.
print("Step-by-step preprocessing:")

# Print the preprocessing table without pandas index numbers.
print(steps.to_string(index=False))

# Print a blank line to separate output sections.
print()

# Print the final cleaned text.
print("Final cleaned text:")

# Print the final cleaned text value.
print(final_text)

# Print a blank line to separate output sections.
print()

# Print the final cleaned token list.
print("Final cleaned tokens:")

# Print the final tokens after stop-word removal.
print(tokens_without_stopwords)

# Print a blank line to separate output sections.
print()

# Print the Bag-of-Words term-frequency table heading.
print("Bag-of-Words term frequencies:")

# Print the term-frequency table without pandas index numbers.
print(term_frequencies.to_string(index=False))

# Print a blank line to separate output sections.
print()

# Print the Bag-of-Words vector table heading.
print("Bag-of-Words vector:")

# Print the Bag-of-Words vector table without pandas index numbers.
print(bow_df.to_string(index=False))

# Print a blank line to separate output sections.
print()

# Print the TF-IDF table heading.
print("TF-IDF representation for sample posts:")

# Print the TF-IDF table rounded to three decimal places.
print(tfidf_df.round(3).to_string(index=False))

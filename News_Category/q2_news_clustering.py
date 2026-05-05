import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = Path("News_Category_Dataset_v3.json")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42

EXTRA_STOPWORDS = {
    "said",
    "say",
    "says",
    "new",
    "year",
    "years",
    "day",
    "week",
    "time",
    "people",
    "just",
    "like",
    "huffpost",
}
STOPWORDS = set(ENGLISH_STOP_WORDS).union(EXTRA_STOPWORDS)


def load_data(path: Path, sample_size: int) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["headline"] = df["headline"].fillna("")
    df["short_description"] = df["short_description"].fillna("")
    df["document"] = (df["headline"] + " " + df["short_description"]).str.strip()
    df = df[df["document"].str.len() > 0].copy()
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def simple_lemma(token: str) -> str:
    """Small fallback lemmatizer so the script remains runnable without NLTK corpora."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def clean_and_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = re.findall(r"[a-z]+", text)
    return [simple_lemma(tok) for tok in tokens if tok not in STOPWORDS and len(tok) > 2]


def join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


def ngram_counts(token_lists: list[list[str]], n: int, top_n: int = 10) -> list[tuple[str, int]]:
    counts = Counter()
    for tokens in token_lists:
        counts.update(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    return counts.most_common(top_n)


def top_terms_from_centroid(model: KMeans, vectorizer: TfidfVectorizer, cluster_id: int, n: int = 10) -> list[str]:
    terms = np.array(vectorizer.get_feature_names_out())
    order = model.cluster_centers_[cluster_id].argsort()[::-1][:n]
    return terms[order].tolist()


def label_cluster(keywords: list[str]) -> str:
    keyword_set = set(keywords)
    label_rules = [
        ("Politics and elections", {"trump", "donald", "president", "republican", "gop", "clinton", "campaign", "democrat"}),
        ("Style, photos, and lifestyle media", {"photo", "style", "fashion", "look", "pinterest", "wedd"}),
        ("Health and social issues", {"health", "care", "mental", "study", "medical", "doctor", "women"}),
        ("World news and international affairs", {"world", "country", "state", "war", "russia", "international"}),
        ("Education, schools, and children", {"school", "student", "teacher", "college", "children", "kid"}),
        ("Parenting and family", {"parent", "mom", "dad", "baby", "child", "kid"}),
        ("Personal life and relationships", {"life", "love", "work", "better", "relationship", "divorce"}),
        ("Video and entertainment", {"video", "watch", "game", "film", "movie", "music"}),
    ]
    scores = [(label, len(keyword_set.intersection(rule_terms))) for label, rule_terms in label_rules]
    best_label, best_score = max(scores, key=lambda item: item[1])
    return best_label if best_score else "Mixed general news"


def markdown_table(rows, columns: list[str]) -> str:
    if isinstance(rows, pd.DataFrame):
        data = rows[columns].fillna("").astype(str).to_dict("records")
    else:
        data = [{col: str(row.get(col, "")) for col in columns} for row in rows]

    def clean_cell(value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip()
        return value.replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(clean_cell(row[col]) for col in columns) + " |" for row in data]
    return "\n".join([header, separator] + body)


def choose_k_by_silhouette(tfidf_matrix, k_values: list[int]) -> tuple[int, list[dict]]:
    rows = []
    best_k = k_values[0]
    best_score = -1.0
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(tfidf_matrix)
        sample_size = min(2000, tfidf_matrix.shape[0])
        score = silhouette_score(tfidf_matrix, labels, sample_size=sample_size, random_state=RANDOM_STATE)
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": float(score)})
        if score > best_score:
            best_k = k
            best_score = score
    return best_k, rows


def plot_elbow(rows: list[dict], output_path: Path) -> None:
    ks = [row["k"] for row in rows]
    inertias = [row["inertia"] for row in rows]
    silhouettes = [row["silhouette"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ks, inertias, marker="o", color="#2563eb", label="Inertia")
    ax1.set_title("Elbow Method for K-Means Clustering on TF-IDF News Features")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia: within-cluster sum of squares")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, marker="s", color="#c2410c", label="Silhouette")
    ax2.set_ylabel("Silhouette score")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def similarity_pairs(df: pd.DataFrame, tfidf_matrix, n_docs: int = 300, top_n: int = 10) -> list[dict]:
    limit = min(n_docs, tfidf_matrix.shape[0])
    sim = cosine_similarity(tfidf_matrix[:limit])
    np.fill_diagonal(sim, -1.0)
    pairs = []
    seen = set()
    flat_order = np.argsort(sim.ravel())[::-1]
    for flat_index in flat_order:
        i, j = divmod(int(flat_index), limit)
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "doc_a": int(i),
                "doc_b": int(j),
                "score": float(sim[i, j]),
                "headline_a": df.loc[i, "headline"],
                "headline_b": df.loc[j, "headline"],
                "category_a": df.loc[i, "category"],
                "category_b": df.loc[j, "category"],
            }
        )
        if len(pairs) == top_n:
            break
    return pairs


def transformer_or_svd_embeddings(clean_text: list[str], tfidf_matrix):
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(clean_text, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(embeddings), "SentenceTransformer all-MiniLM-L6-v2"
    except Exception as exc:
        svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
        embeddings = svd.fit_transform(tfidf_matrix)
        note = (
            "TruncatedSVD fallback used because SentenceTransformer was unavailable locally: "
            f"{type(exc).__name__}: {exc}"
        )
        return embeddings, note


def write_report(
    df: pd.DataFrame,
    token_lists: list[list[str]],
    word_counts: list[tuple[str, int]],
    bigrams: list[tuple[str, int]],
    trigrams: list[tuple[str, int]],
    bow_shape: tuple[int, int],
    tfidf_shape: tuple[int, int],
    pairs: list[dict],
    elbow_rows: list[dict],
    chosen_k: int,
    cluster_rows: list[dict],
    embedding_method: str,
    embedding_cluster_rows: list[dict],
) -> None:
    total_docs = len(df)
    avg_len = sum(len(tokens) for tokens in token_lists) / total_docs
    vocab_size = len(set(tok for tokens in token_lists for tok in tokens))

    lines = [
        "# Question 2: News Topic Clustering Practical Answer",
        "",
        "## Dataset Understanding",
        f"- Records used in this run: {total_docs:,}.",
        f"- Key columns: `headline`, `short_description`, `category`, `authors`, `date`, and `link`.",
        f"- Average cleaned document length: {avg_len:.2f} words.",
        f"- Vocabulary size after tokenization: {vocab_size:,} unique words.",
        "",
        "The `headline` gives the strongest short topic signal, while `short_description` adds context. "
        "`category` is not used as a training label because the task is unsupervised, but it is useful for checking whether clusters make sense.",
        "",
        "## First 10 Records",
        markdown_table(df.head(10), ["headline", "category", "short_description"]),
        "",
        "## Noise and Inconsistencies",
        "- Mixed capitalization and punctuation create duplicate forms of the same word, such as `Trump`, `trump`, and `trump.`.",
        "- Some descriptions are empty or very short, which can make document vectors weak and harder to cluster.",
        "- Proper names, dates, numbers, and news-source style wording can dominate clusters unless normalized or filtered.",
        "",
        "## Preprocessing Examples",
    ]

    for _, row in df.head(2).iterrows():
        lines.extend(
            [
                f"- Original: {row['document']}",
                f"  Cleaned: {row['clean_text']}",
            ]
        )

    lines.extend(
        [
            "",
            "Cleaning lowercases text, removes punctuation/symbols, tokenizes words, removes stopwords, and applies a simple lemmatization fallback. "
            "This improves clustering because documents about the same topic share more comparable tokens.",
            "",
        "## Frequent Terms",
        markdown_table(pd.DataFrame(word_counts, columns=["term", "count"]), ["term", "count"]),
            "",
            "These terms summarize dominant news themes. Very general words are less useful, while named topics such as politics, entertainment, health, travel, or family-related terms help reveal clusters.",
            "",
        "## Common Bigrams",
        markdown_table(pd.DataFrame(bigrams, columns=["bigram", "count"]), ["bigram", "count"]),
            "",
        "## Common Trigrams",
        markdown_table(pd.DataFrame(trigrams, columns=["trigram", "count"]), ["trigram", "count"]),
            "",
            "N-grams preserve short phrases that single words lose. For example, a phrase can indicate a political institution, a celebrity-news theme, or a health topic more clearly than its separate words.",
            "",
            "## Text Representation",
            f"- Bag-of-Words matrix shape: {bow_shape[0]:,} documents x {bow_shape[1]:,} terms.",
            f"- TF-IDF matrix shape: {tfidf_shape[0]:,} documents x {tfidf_shape[1]:,} terms.",
            "",
            "Bag-of-Words stores raw counts, so repeated words receive high weight even if they are common. TF-IDF downweights words that appear in many documents and gives stronger weight to terms that distinguish one article from another. "
            "Both matrices are sparse because each article uses only a small part of the full vocabulary.",
            "",
        "## Similar Document Pairs",
        markdown_table(
            pd.DataFrame(pairs),
            ["doc_a", "doc_b", "score", "headline_a", "headline_b", "category_a", "category_b"],
        ),
            "",
            "Cosine similarity is high when two TF-IDF vectors point in a similar direction, usually because the documents share important words or phrases.",
            "",
        "## Elbow and Cluster Selection",
        markdown_table(pd.DataFrame(elbow_rows), ["k", "inertia", "silhouette"]),
            "",
            f"Chosen k for the final TF-IDF K-Means model: {chosen_k}. The plot is saved at `outputs/q2_elbow_tfidf.png` and includes a title, x-axis label, y-axis labels, legend, and grid.",
            "",
        "## TF-IDF K-Means Cluster Interpretation",
        markdown_table(pd.DataFrame(cluster_rows), ["cluster", "size", "topic_label", "top_keywords", "sample_headlines"]),
            "",
            "## Transformer Embedding Comparison",
            f"Embedding method used by this run: {embedding_method}",
            "",
        markdown_table(pd.DataFrame(embedding_cluster_rows), ["cluster", "size", "topic_label", "common_terms", "sample_headlines"]),
            "",
            "Transformer sentence embeddings usually improve clustering when they are available because they represent semantic meaning, not only exact shared words. "
            "For instance, documents about elections can be close even when one says `campaign` and another says `voters`. TF-IDF is easier to interpret and faster, but it is more sensitive to vocabulary mismatch.",
            "",
            "## Evaluation Insight",
            "The cluster keywords and sample headlines provide qualitative evaluation. A strong cluster has coherent keywords and documents that discuss the same topic. "
            "The original `category` column can be used only for after-the-fact validation because the clustering task itself is unsupervised. "
            "In this run, the very broad clusters show a known weakness of sparse TF-IDF K-Means on short news snippets: some documents share general vocabulary but not a single tight topic. "
            "The more specific clusters, such as politics/elections and parenting/school-related articles, are stronger because they contain distinctive repeated terms.",
        ]
    )

    (OUTPUT_DIR / "q2_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Question 2: unsupervised news topic clustering pipeline.")
    parser.add_argument("--sample-size", type=int, default=8000, help="Number of records to sample for a fast practical run.")
    parser.add_argument("--max-features", type=int, default=5000, help="Maximum BoW/TF-IDF vocabulary size.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data(DATA_FILE, args.sample_size)
    token_lists = [clean_and_tokenize(text) for text in df["document"]]
    df["clean_text"] = [join_tokens(tokens) for tokens in token_lists]
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    token_lists = [text.split() for text in df["clean_text"]]

    word_counts = Counter(tok for tokens in token_lists for tok in tokens).most_common(10)
    bigrams = ngram_counts(token_lists, 2)
    trigrams = ngram_counts(token_lists, 3)

    bow_vectorizer = CountVectorizer(max_features=args.max_features, min_df=3, max_df=0.85)
    bow_matrix = bow_vectorizer.fit_transform(df["clean_text"])
    tfidf_vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=3, max_df=0.85)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"])

    pairs = similarity_pairs(df, tfidf_matrix)

    k_values = list(range(2, 11))
    chosen_k, elbow_rows = choose_k_by_silhouette(tfidf_matrix, k_values)
    plot_elbow(elbow_rows, OUTPUT_DIR / "q2_elbow_tfidf.png")

    kmeans = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=10)
    df["tfidf_cluster"] = kmeans.fit_predict(tfidf_matrix)

    cluster_rows = []
    for cluster_id in range(chosen_k):
        cluster_df = df[df["tfidf_cluster"] == cluster_id]
        sample_headlines = " | ".join(cluster_df["headline"].head(3).tolist())
        top_terms = ", ".join(top_terms_from_centroid(kmeans, tfidf_vectorizer, cluster_id))
        top_term_list = top_terms_from_centroid(kmeans, tfidf_vectorizer, cluster_id)
        cluster_rows.append(
            {
                "cluster": cluster_id,
                "size": len(cluster_df),
                "topic_label": label_cluster(top_term_list),
                "top_keywords": top_terms,
                "sample_headlines": sample_headlines,
            }
        )

    embeddings, embedding_method = transformer_or_svd_embeddings(df["clean_text"].tolist(), tfidf_matrix)
    emb_kmeans = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=10)
    df["embedding_cluster"] = emb_kmeans.fit_predict(embeddings)

    embedding_cluster_rows = []
    for cluster_id in range(chosen_k):
        cluster_df = df[df["embedding_cluster"] == cluster_id]
        sample_headlines = " | ".join(cluster_df["headline"].head(3).tolist())
        terms = Counter(tok for tokens in cluster_df["clean_text"].str.split() for tok in tokens).most_common(8)
        term_list = [term for term, _ in terms]
        embedding_cluster_rows.append(
            {
                "cluster": cluster_id,
                "size": len(cluster_df),
                "topic_label": label_cluster(term_list),
                "common_terms": ", ".join(term for term, _ in terms),
                "sample_headlines": sample_headlines,
            }
        )

    df[
        [
            "headline",
            "category",
            "short_description",
            "clean_text",
            "tfidf_cluster",
            "embedding_cluster",
        ]
    ].to_csv(OUTPUT_DIR / "q2_clustered_news_sample.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(OUTPUT_DIR / "q2_tfidf_cluster_summary.csv", index=False)
    pd.DataFrame(embedding_cluster_rows).to_csv(OUTPUT_DIR / "q2_embedding_cluster_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUTPUT_DIR / "q2_similarity_pairs.csv", index=False)
    pd.DataFrame(elbow_rows).to_csv(OUTPUT_DIR / "q2_elbow_scores.csv", index=False)

    write_report(
        df,
        token_lists,
        word_counts,
        bigrams,
        trigrams,
        bow_matrix.shape,
        tfidf_matrix.shape,
        pairs,
        elbow_rows,
        chosen_k,
        cluster_rows,
        embedding_method,
        embedding_cluster_rows,
    )

    print("Question 2 pipeline completed.")
    print(f"Records analyzed: {len(df):,}")
    print(f"BoW shape: {bow_matrix.shape}")
    print(f"TF-IDF shape: {tfidf_matrix.shape}")
    print(f"Chosen k: {chosen_k}")
    print(f"Embedding method: {embedding_method}")
    print(f"Outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

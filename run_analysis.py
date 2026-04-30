"""
Ryanair Trustpilot reviews - Text analysis pipeline.

Five visualisations are produced:
  fig_01_rating_distribution.png   Rating distribution
  fig_02_bow_top20.png             Bag-of-Words top 20 unigrams
  fig_03_tfidf_top20.png           TF-IDF top 20 unigrams
  fig_04_lda_topics.png            LDA topic words (5 topics)
  fig_05_sentiment_confusion.png   Sentiment classifier confusion matrix
  fig_06_review_volume_time.png    Review volume over time
  fig_07_bigrams.png               Top bigrams (BoW)

Outputs saved to /home/ryanair_data/figs/
"""

import re
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")
PALETTE_RYANAIR = ["#073590", "#F1C40F", "#E74C3C", "#1ABC9C", "#9B59B6"]

DATA_DIR = Path("/home/ryanair_data")
FIG_DIR = DATA_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD CORPUS
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "ryanair_corpus.csv")
print(f"Loaded {len(df)} reviews")
print(df["rating"].value_counts().sort_index())

# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------
# Stop words list - based on standard English stopwords + a few domain-specific
# tokens that swamp the analysis without adding meaning ("ryanair", "flight").
STOPWORDS = set("""
a about above after again against all am an and any are aren as at be because
been before being below between both but by can cannot could couldn did didn
do does doesn doing don down during each few for from further had hadn has
hasn have haven having he her here hers herself him himself his how i if in
into is isn it its itself just ll let me might more most mustn my myself no
nor not now of off on once only or other ought our ours ourselves out over
own re s same shan she should shouldn so some such t than that the their
theirs them themselves then there these they this those through to too under
until up ve very was wasn we were weren what when where which while who whom
why will with won would wouldn you your yours yourself yourselves get got
also even one two three really would still get got like make made also way
much many lot back even though although however thus therefore well say said
think thought come came went going go know knew tell told see saw take took
flew fly flying flight flights ryanair air ryan airline airlines
""".split())


def preprocess(text):
    text = str(text).lower()
    # remove urls and emails
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    # remove digits and punctuation
    text = re.sub(r"[\d]", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # tokenise + stopword removal + length filter
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


df["clean"] = df["review"].apply(preprocess)
df["n_words"] = df["clean"].str.split().str.len()
print(f"\nMean cleaned-review length (words): {df['n_words'].mean():.1f}")
print(f"Median cleaned-review length (words): {df['n_words'].median():.0f}")

# Drop empty after preprocessing
df = df[df["n_words"] >= 3].reset_index(drop=True)
print(f"Reviews after preprocessing filter: {len(df)}")

# ---------------------------------------------------------------------------
# 3. RATING DISTRIBUTION (Figure 1)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
counts = df["rating"].value_counts().sort_index()
prop = (counts / counts.sum() * 100).round(1)
bars = ax.bar(counts.index.astype(str), counts.values,
              color=["#C0392B", "#E67E22", "#F1C40F", "#27AE60", "#2ECC71"],
              edgecolor="black", linewidth=0.6)
for bar, p, c in zip(bars, prop.values, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f"{c}\n({p}%)", ha="center", fontsize=9)
ax.set_xlabel("Star rating")
ax.set_ylabel("Number of reviews")
ax.set_title("Figure 1. Distribution of Trustpilot star ratings for Ryanair (n = {:,})".format(len(df)),
             fontsize=11)
ax.set_ylim(0, counts.max() * 1.15)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_01_rating_distribution.png", bbox_inches="tight")
plt.close()
print("Saved Figure 1")

# ---------------------------------------------------------------------------
# 4. BAG-OF-WORDS (Figure 2)
# ---------------------------------------------------------------------------
bow = CountVectorizer(max_features=2000, ngram_range=(1, 1), min_df=3)
X_bow = bow.fit_transform(df["clean"])
vocab = bow.get_feature_names_out()
freq = X_bow.sum(axis=0).A1
freq_series = pd.Series(freq, index=vocab).sort_values(ascending=False)
top20 = freq_series.head(20)

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=top20.values, y=top20.index, ax=ax,
            palette=sns.color_palette("crest_r", n_colors=20))
for i, v in enumerate(top20.values):
    ax.text(v + 5, i, str(int(v)), va="center", fontsize=8)
ax.set_xlabel("Document frequency")
ax.set_ylabel("Token")
ax.set_title("Figure 2. Top 20 unigrams by raw document frequency (BoW)", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_02_bow_top20.png", bbox_inches="tight")
plt.close()
print("Saved Figure 2")

# Save top words for the report tables
top20.to_csv(DATA_DIR / "top20_bow.csv", header=["frequency"])

# ---------------------------------------------------------------------------
# 5. TF-IDF (Figure 3)
# ---------------------------------------------------------------------------
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 1), min_df=3,
                        sublinear_tf=True)
X_tfidf = tfidf.fit_transform(df["clean"])
tfidf_vocab = tfidf.get_feature_names_out()
# Take the per-token MAX TF-IDF score across the corpus - this surfaces
# tokens that are highly distinctive in at least one review, mirroring
# the c_v-style approach demonstrated in lectures.
max_scores = X_tfidf.max(axis=0).toarray().flatten()
mean_scores = X_tfidf.mean(axis=0).A1
score_df = pd.DataFrame({"token": tfidf_vocab, "max_tfidf": max_scores,
                         "mean_tfidf": mean_scores})
score_df = score_df.sort_values("mean_tfidf", ascending=False)
top20_tfidf = score_df.head(20)

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x="mean_tfidf", y="token", data=top20_tfidf, ax=ax,
            palette=sns.color_palette("rocket_r", n_colors=20))
for i, v in enumerate(top20_tfidf["mean_tfidf"].values):
    ax.text(v + 0.0005, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("Mean TF-IDF weight across corpus")
ax.set_ylabel("Token")
ax.set_title("Figure 3. Top 20 unigrams by mean TF-IDF weight", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_03_tfidf_top20.png", bbox_inches="tight")
plt.close()
print("Saved Figure 3")
top20_tfidf.to_csv(DATA_DIR / "top20_tfidf.csv", index=False)

# ---------------------------------------------------------------------------
# 6. LDA TOPIC MODELLING (Figure 4)
# ---------------------------------------------------------------------------
N_TOPICS = 5
lda_vectoriser = CountVectorizer(max_features=2000, min_df=5, max_df=0.7,
                                 ngram_range=(1, 2))
X_lda = lda_vectoriser.fit_transform(df["clean"])
print(f"LDA input matrix: {X_lda.shape}")
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42,
                                max_iter=30, learning_method="batch")
lda.fit(X_lda)
feat = lda_vectoriser.get_feature_names_out()

# Pull top 10 terms per topic
TOP_N = 10
topic_terms = []
for k, comp in enumerate(lda.components_):
    top_idx = comp.argsort()[::-1][:TOP_N]
    terms = [(feat[i], comp[i]) for i in top_idx]
    topic_terms.append(terms)

# Topic labels - assigned manually after inspecting the LDA output. Because
# the corpus is dominated by negative reviews, several topics share similar
# vocabulary; the labels below distinguish them by their dominant lexical
# signal (e.g. Topic 0 leads with 'charged'+'sizer'+'fit', Topic 4 leads
# with 'compensation'+'hours'+'delayed').
TOPIC_LABEL_RULES = [
    # ordered: pick the FIRST rule that matches; each topic gets a unique label
    ("Baggage and gate-side fees",
     ["sizer", "charged pounds", "fit", "bag"]),
    ("Bookings, hold luggage and ancillary services",
     ["booking", "car", "hold", "booked"]),
    ("Hidden fees and the cheap-flight trap",
     ["cheap", "fees", "clearly", "claim", "app"]),
    ("Online check-in and boarding-pass charges",
     ["online check", "boarding", "deliberately", "online"]),
    ("Delays, refunds and compensation claims",
     ["compensation", "delayed", "delayed hours", "refund", "hours"]),
]

def label_topic(terms_with_weights, used):
    """Assign the highest-priority unused label that has a keyword match."""
    words = [t for t, _ in terms_with_weights]
    word_set = set(words)
    # Score each unused rule by number of keyword hits in this topic
    scores = []
    for label, keywords in TOPIC_LABEL_RULES:
        if label in used:
            continue
        hits = sum(1 for kw in keywords if any(kw in w or w in kw for w in word_set))
        scores.append((hits, label))
    scores.sort(reverse=True)
    if scores and scores[0][0] > 0:
        return scores[0][1]
    # fallback: first unused label
    for label, _ in TOPIC_LABEL_RULES:
        if label not in used:
            return label
    return "General complaint"

# Assign in order of topic ID so each gets a unique label
used_labels = set()
labels = []
for k, terms in enumerate(topic_terms):
    lab = label_topic(terms, used_labels)
    labels.append(lab)
    used_labels.add(lab)
print("\nLDA topic labels:")
for k, lab in enumerate(labels):
    print(f"  Topic {k}: {lab}")
    print(f"    Top terms: {[t for t, _ in topic_terms[k]]}")

fig, axes = plt.subplots(1, N_TOPICS, figsize=(20, 6))
for k, ax in enumerate(axes):
    terms, weights = zip(*topic_terms[k])
    ax.barh(range(len(terms)), weights, color=PALETTE_RYANAIR[k % len(PALETTE_RYANAIR)],
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"Topic {k}\n{labels[k]}", fontsize=10)
    ax.set_xlabel("Term weight")
fig.suptitle(f"Figure 4. LDA topic model (K={N_TOPICS}) - top terms per topic",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_04_lda_topics.png", bbox_inches="tight")
plt.close()
print("Saved Figure 4")

# Topic diversity score (per Week 9 lecture handout)
all_top_words = [w for terms in topic_terms for w, _ in terms]
diversity = len(set(all_top_words)) / len(all_top_words)
print(f"\nLDA topic diversity (Dieng et al., 2020): {diversity:.3f}")

# Save topic terms table
with open(DATA_DIR / "lda_topics.txt", "w") as f:
    f.write(f"LDA topic model (K={N_TOPICS}, top {TOP_N} terms per topic)\n")
    f.write(f"Topic diversity: {diversity:.3f}\n\n")
    for k, lab in enumerate(labels):
        f.write(f"Topic {k} - {lab}\n")
        for t, w in topic_terms[k]:
            f.write(f"  {t:<30} {w:.4f}\n")
        f.write("\n")

# ---------------------------------------------------------------------------
# 7. SENTIMENT CLASSIFICATION (Figure 5)
# ---------------------------------------------------------------------------
# Map star rating -> sentiment label (the standard mapping used by
# Pang & Lee 2008 and most subsequent review-sentiment studies)
def to_sentiment(r):
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"

df["sentiment"] = df["rating"].apply(to_sentiment)
print(df["sentiment"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["sentiment"], test_size=0.20, stratify=df["sentiment"],
    random_state=42)

# Naive Bayes
nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)),
    ("clf", MultinomialNB(alpha=0.1)),
])
nb_pipe.fit(X_train, y_train)
nb_pred = nb_pipe.predict(X_test)

# Logistic Regression
lr_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3)),
    ("clf", LogisticRegression(C=4.0, max_iter=1000, class_weight="balanced",
                               random_state=42)),
])
lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)

print("\nNaive Bayes")
print(f"  Accuracy : {accuracy_score(y_test, nb_pred):.3f}")
print(f"  Macro F1 : {f1_score(y_test, nb_pred, average='macro'):.3f}")
print(classification_report(y_test, nb_pred, zero_division=0))

print("Logistic Regression")
print(f"  Accuracy : {accuracy_score(y_test, lr_pred):.3f}")
print(f"  Macro F1 : {f1_score(y_test, lr_pred, average='macro'):.3f}")
print(classification_report(y_test, lr_pred, zero_division=0))

# Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
labels_order = ["negative", "neutral", "positive"]
for ax, preds, name in [(axes[0], nb_pred, "Naive Bayes"),
                        (axes[1], lr_pred, "Logistic Regression")]:
    cm = confusion_matrix(y_test, preds, labels=labels_order)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=cm, fmt="d", cmap="Blues",
                xticklabels=labels_order, yticklabels=labels_order, ax=ax,
                cbar_kws={"label": "Row %"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")
    ax.set_title(f"{name}\nAccuracy={acc:.3f}  Macro-F1={f1m:.3f}")
fig.suptitle("Figure 5. Sentiment classifier confusion matrices (held-out test set, 20%)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_05_sentiment_confusion.png", bbox_inches="tight")
plt.close()
print("Saved Figure 5")

# Save classifier metrics
metrics = {
    "Naive Bayes": {
        "accuracy": float(accuracy_score(y_test, nb_pred)),
        "macro_f1": float(f1_score(y_test, nb_pred, average="macro")),
    },
    "Logistic Regression": {
        "accuracy": float(accuracy_score(y_test, lr_pred)),
        "macro_f1": float(f1_score(y_test, lr_pred, average="macro")),
    },
}
with open(DATA_DIR / "classifier_metrics.txt", "w") as f:
    for name, m in metrics.items():
        f.write(f"{name}: accuracy={m['accuracy']:.3f}, macro_f1={m['macro_f1']:.3f}\n")

# ---------------------------------------------------------------------------
# 8. REVIEW VOLUME OVER TIME (Figure 6)
# ---------------------------------------------------------------------------
df["dt"] = pd.to_datetime(df["date"], errors="coerce")
ts_df = df.dropna(subset=["dt"]).copy()
ts_df["month"] = ts_df["dt"].dt.to_period("M").dt.to_timestamp()
monthly = ts_df.groupby(["month", "sentiment"]).size().unstack(fill_value=0)
# ensure all sentiments present
for col in ["negative", "neutral", "positive"]:
    if col not in monthly.columns:
        monthly[col] = 0
monthly = monthly[["negative", "neutral", "positive"]]

fig, ax = plt.subplots(figsize=(11, 5))
monthly.plot(kind="area", stacked=True, ax=ax,
             color=["#C0392B", "#F1C40F", "#27AE60"], alpha=0.85)
ax.set_xlabel("Month")
ax.set_ylabel("Number of reviews")
ax.set_title("Figure 6. Monthly review volume by sentiment (Sep 2024 - Apr 2026)",
             fontsize=11)
ax.legend(title="Sentiment", loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_06_review_volume_time.png", bbox_inches="tight")
plt.close()
print("Saved Figure 6")

# ---------------------------------------------------------------------------
# 9. TOP BIGRAMS (Figure 7)
# ---------------------------------------------------------------------------
bow_bi = CountVectorizer(max_features=2000, ngram_range=(2, 2), min_df=5)
X_bi = bow_bi.fit_transform(df["clean"])
bi_vocab = bow_bi.get_feature_names_out()
bi_freq = pd.Series(X_bi.sum(axis=0).A1, index=bi_vocab).sort_values(ascending=False)
top15_bi = bi_freq.head(15)

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=top15_bi.values, y=top15_bi.index, ax=ax,
            palette=sns.color_palette("flare_r", n_colors=15))
for i, v in enumerate(top15_bi.values):
    ax.text(v + 2, i, str(int(v)), va="center", fontsize=8)
ax.set_xlabel("Document frequency")
ax.set_ylabel("Bigram")
ax.set_title("Figure 7. Top 15 bigrams (consecutive word pairs)", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_07_bigrams.png", bbox_inches="tight")
plt.close()
print("Saved Figure 7")
top15_bi.to_csv(DATA_DIR / "top15_bigrams.csv", header=["frequency"])

# ---------------------------------------------------------------------------
# 10. SUMMARY STATS FOR THE REPORT
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY STATS FOR REPORT")
print("=" * 60)
print(f"Total reviews                 : {len(df):,}")
print(f"Mean rating                   : {df['rating'].mean():.2f}")
print(f"% 1-star                      : {(df['rating']==1).mean()*100:.1f}%")
print(f"% 5-star                      : {(df['rating']==5).mean()*100:.1f}%")
print(f"Mean cleaned-review length    : {df['n_words'].mean():.1f} words")
print(f"Vocabulary size (BoW)         : {len(vocab):,}")
print(f"LDA topic diversity           : {diversity:.3f}")
print(f"NB accuracy / macro-F1        : {metrics['Naive Bayes']['accuracy']:.3f} / {metrics['Naive Bayes']['macro_f1']:.3f}")
print(f"LR accuracy / macro-F1        : {metrics['Logistic Regression']['accuracy']:.3f} / {metrics['Logistic Regression']['macro_f1']:.3f}")

# Save the analysed dataframe for downstream use
df.to_csv(DATA_DIR / "ryanair_corpus_analysed.csv", index=False)
print("\nAll outputs saved to", FIG_DIR)

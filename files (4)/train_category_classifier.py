"""
train_category_classifier.py

Trains a free, local resume-category classifier on data/Resume.csv
(the Kaggle "Resume Dataset" — 2,484 resumes across 24 job categories).

This is a real TF-IDF + Linear SVM text classifier, not an LLM call — it
costs nothing to run and nothing to use afterward, and it's what actually
fits this dataset: category labels, not job-description match scores.
An LLM fine-tune would need score labels we don't have; this is the
correct tool for the data we actually have.

Usage:
    python train_category_classifier.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "data/Resume.csv"
MODEL_PATH = "category_classifier.pkl"


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Resume_str", "Category"])
    print(f"Loaded {len(df)} resumes across {df['Category'].nunique()} categories")

    X = df["Resume_str"]
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
        )),
        ("clf", LinearSVC(random_state=42)),
    ])

    print("\nTraining...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()

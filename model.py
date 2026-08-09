import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

import joblib


INPUT_FILE = "labeled_fast.csv"
MODEL_PATH = "email_classifier.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def load_data():
    df = pd.read_csv(INPUT_FILE)

    df = df[df["category"].notna()]

    # combine text
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

    return df


def prepare_features(df):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(df["text"])
    y = df["category"]

    return X, y, vectorizer


def train_models(X_train, y_train):

    # Logistic Regression
    model1 = LogisticRegression(max_iter=1000, class_weight="balanced")

    # Naive Bayes
    model2 = MultinomialNB()

    # Random Forest
    model3 = RandomForestClassifier(n_estimators=100, class_weight="balanced")

    # XGBoost
    model4 = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    # LinearSVC → wrapped for probabilities
    svc = LinearSVC()
    model5 = CalibratedClassifierCV(svc)

    ensemble = VotingClassifier(
        estimators=[
            ("lr", model1),
            ("nb", model2),
            ("rf", model3),
            ("xgb", model4),
            ("svc", model5)
        ],
        voting="soft"
    )

    ensemble.fit(X_train, y_train)

    return ensemble


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


def main():
    print("Loading dataset...")
    df = load_data()

    print("Preparing features...")
    X, y, vectorizer = prepare_features(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models...")
    model = train_models(X_train, y_train)

    print("Evaluating model...")
    evaluate(model, X_test, y_test)

    print("Saving model...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nDone! Model ready.")


if __name__ == "__main__":
    main()
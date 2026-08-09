import argparse

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


INPUT_FILE = "labeled_fast.csv"
MODEL_PATH = "email_classifier.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"


def load_data(input_file=INPUT_FILE):
    df = pd.read_csv(input_file)
    df = df[df["category"].notna()].copy()
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    return df


def prepare_features(df):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(df["text"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    return X, y, vectorizer, label_encoder


def train_models(X_train, y_train):
    model1 = LogisticRegression(max_iter=1000, class_weight="balanced")
    model2 = MultinomialNB()
    model3 = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )
    model4 = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric="mlogloss",
        objective="multi:softprob",
    )
    model5 = CalibratedClassifierCV(LinearSVC(dual="auto"), cv=3)

    ensemble = VotingClassifier(
        estimators=[
            ("lr", model1),
            ("nb", model2),
            ("rf", model3),
            ("xgb", model4),
            ("svc", model5),
        ],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)
    return ensemble


def evaluate(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )


def main(input_file=INPUT_FILE):
    print("Loading dataset...")
    df = load_data(input_file)
    print(f"Rows: {len(df)} | Classes: {sorted(df['category'].unique())}")

    print("Preparing features...")
    X, y, vectorizer, label_encoder = prepare_features(df)

    print("Splitting dataset...")
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    print("Training models...")
    model = train_models(X_train, y_train)

    print("Evaluating model...")
    evaluate(model, X_test, y_test, label_encoder)

    print("Saving model...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print("\nDone! Model ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train email classifier")
    parser.add_argument("--input", default=INPUT_FILE, help="Labeled CSV path")
    args = parser.parse_args()
    main(input_file=args.input)

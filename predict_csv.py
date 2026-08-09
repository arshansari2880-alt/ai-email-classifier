import pandas as pd
import joblib

INPUT_FILE = "input.csv"
OUTPUT_FILE = "output_predictions.csv"

MODEL_PATH = "email_classifier.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def main():
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)

    # Handle missing values
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    print("Preparing text...")
    df["text"] = df["subject"] + " " + df["body"]

    print("Transforming...")
    X = vectorizer.transform(df["text"])

    print("Predicting...")
    df["predicted_category"] = model.predict(X)

    # Optional: confidence (probability)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        df["confidence"] = probs.max(axis=1)

    print("Saving output...")
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Done! Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
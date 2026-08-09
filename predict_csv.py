import argparse

import joblib
import pandas as pd


INPUT_FILE = "input.csv"
OUTPUT_FILE = "output_predictions.csv"
MODEL_PATH = "email_classifier.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"


def main(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = None
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except FileNotFoundError:
        pass

    print("Loading data...")
    df = pd.read_csv(input_file)
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    print("Preparing text...")
    text = df["subject"] + " " + df["body"]

    print("Transforming...")
    X = vectorizer.transform(text)

    print("Predicting...")
    preds = model.predict(X)
    if label_encoder is not None:
        df["predicted_category"] = label_encoder.inverse_transform(preds)
    else:
        df["predicted_category"] = preds

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        df["confidence"] = probs.max(axis=1).round(4)

    print("Saving output...")
    df.to_csv(output_file, index=False)
    print(f"Done! Predictions saved to {output_file}")
    print(df[["subject", "predicted_category"] + (["confidence"] if "confidence" in df.columns else [])].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict email categories from CSV")
    parser.add_argument("--input", default=INPUT_FILE, help="Input CSV with subject/body")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output CSV path")
    args = parser.parse_args()
    main(input_file=args.input, output_file=args.output)

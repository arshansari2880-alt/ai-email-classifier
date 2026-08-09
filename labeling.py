import argparse
import os
import re

import ftfy
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


INPUT_CANDIDATES = ["labeled_dataset.csv", "emails_dataset.csv"]
OUTPUT_FILE = "processed_dataset.csv"
LABELED_FAST_FILE = "labeled_fast.csv"

USE_LLM = False
CONF_THRESHOLD = 0.3


PHISHING_PATTERNS = [
    "verify your account", "reset your password", "login to your account",
    "click here to login", "confirm your identity", "update your details",
    "your account will be suspended", "unauthorized access detected",
    "security alert", "otp required", "bank verification", "urgent action required",
    "limited slots available",
]

SPAM_PATTERNS = [
    "unsubscribe", "limited time offer", "exclusive deal", "limited slots available",
    "discount", "buy now", "sale ends soon", "click here", "free trial", "win",
    "congratulations", "you have been selected", "earn money",
    "work from home", "apply now",
]

IMPORTANT_PATTERNS = [
    "meeting", "deadline", "interview", "project update", "assignment",
    "submission", "schedule", "discussion", "client", "review", "presentation",
    "action required",
]

TRANSACTIONAL_PATTERNS = [
    "upi", "credited", "debited", "transaction", "bank alert",
    "account", "balance", "payment", "rs.", "inr", "amount",
    "received", "sent", "withdrawn", "deposit", "neft", "imps",
]

ACADEMIC_PATTERNS = [
    "exam", "examination", "submission", "assignment", "schedule", "timetable",
    "deadline", "students", "batch", "semester", "uniform", "verification",
    "document", "lecture", "lab", "attendance", "internal", "external", "practical",
]

PHISHING_REGEX = re.compile("|".join(re.escape(p) for p in PHISHING_PATTERNS), re.I)
SPAM_REGEX = re.compile("|".join(re.escape(p) for p in SPAM_PATTERNS), re.I)
IMPORTANT_REGEX = re.compile("|".join(re.escape(p) for p in IMPORTANT_PATTERNS), re.I)
ACADEMIC_REGEX = re.compile("|".join(re.escape(p) for p in ACADEMIC_PATTERNS), re.I)
TRANSACTION_REGEX = re.compile("|".join(re.escape(p) for p in TRANSACTIONAL_PATTERNS), re.I)


def resolve_input_file(path=None):
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    for candidate in INPUT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No input CSV found. Expected one of: "
        + ", ".join(INPUT_CANDIDATES)
    )


def is_nonsense(text):
    if len(text) < 5:
        return True
    if len(set(text)) <= 2:
        return True

    words = text.split()
    if len(words) == 1 and len(words[0]) > 6:
        return True

    vowels = sum(c in "aeiou" for c in text.lower())
    if vowels / (len(text) + 1e-6) < 0.1:
        return True

    return False


def fix_text_pipeline(text):
    if not text or not isinstance(text, str):
        return ""

    text = ftfy.fix_text(text)
    text = re.sub(r"\b(\S{1,3})( \1){5,}\b", " ", text)
    text = re.sub(r"(Í\s*)+", " ", text)
    text = " ".join(text.split())
    return text.strip()


def classify_email(subject, body, sender):
    text = f"{subject} {body}".lower()

    if is_nonsense(text):
        return "spam", 0.9

    scores = {"spam": 0, "phishing": 0, "important": 0, "ham": 0}

    if "<url>" in text:
        scores["phishing"] += 1
        scores["spam"] += 1

    if any(k in text for k in ["verify", "password", "login", "otp", "bank"]):
        scores["phishing"] += 2

    if PHISHING_REGEX.search(text):
        scores["phishing"] += 3

    if SPAM_REGEX.search(text):
        scores["spam"] += 2

    if text.count("<url>") > 2:
        scores["spam"] += 1

    if any(k in text for k in ["offer", "sale", "deal"]):
        scores["spam"] += 1

    if IMPORTANT_REGEX.search(text):
        scores["important"] += 2

    if ACADEMIC_REGEX.search(text):
        scores["important"] += 2
        scores["spam"] = max(0, scores["spam"] - 1)

    if scores["phishing"] >= 4:
        return "phishing", 0.9

    sender = str(sender).lower()
    if "symbiosis" in sender or "sit" in sender:
        scores["important"] += 2

    if TRANSACTION_REGEX.search(text):
        scores["ham"] += 3

    scores["ham"] += 1

    category = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    max_score = sorted_scores[0]
    confidence = (
        (sorted_scores[0] - sorted_scores[1]) / (max_score + 1e-6)
        if max_score > 0
        else 0.0
    )
    return category, confidence


def local_llm_classify(text):
    prompt = f"spam/ham/phishing/important?\n{text}"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=60,
        )
        output = response.json()["response"].strip().lower()
        for cat in ["spam", "ham", "phishing", "important"]:
            if cat in output:
                return cat
        return "ham"
    except Exception as e:
        print(f"LLM error: {e}")
        return "ham"


def process_row(row):
    subject = row.get("subject", "")
    body = row.get("body", "")
    sender = row.get("sender", "")
    recipient = row.get("recipient", "")
    has_links = row.get("has_links", "")
    labels = row.get("labels", "")

    category, confidence = classify_email(subject, body, sender)
    return {
        "subject": subject,
        "sender": sender,
        "recipient": recipient,
        "body": body,
        "has_links": has_links,
        "labels": labels,
        "category": category,
        "confidence": confidence,
    }


def run_llm_parallel(df, indices, max_workers=5):
    results = {}
    total = len(indices)

    def worker(idx):
        text = str(df.at[idx, "subject"]) + " " + str(df.at[idx, "body"])
        return idx, local_llm_classify(text[:400])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, idx) for idx in indices]
        for count, future in enumerate(as_completed(futures), start=1):
            idx, category = future.result()
            results[idx] = category
            if count <= 5 or count % 50 == 0:
                print(f"LLM Progress: {count}/{total}")

    return results


def label_dataset(input_file=None, use_llm=USE_LLM):
    input_file = resolve_input_file(input_file)
    print(f"Reading: {input_file}")

    df = pd.read_csv(input_file, engine="python", on_bad_lines="skip")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    for col in ("subject", "body", "sender"):
        if col not in df.columns:
            df[col] = ""

    df["subject"] = df["subject"].apply(fix_text_pipeline)
    df["body"] = df["body"].apply(fix_text_pipeline)

    print(f"Total rows: {len(df)}")
    print("Labeling with rules...")

    result_df = pd.DataFrame([process_row(row) for row in df.to_dict("records")])
    result_df.to_csv(LABELED_FAST_FILE, index=False)
    print(f"Saved rule labels to {LABELED_FAST_FILE}")

    if use_llm:
        mask = result_df["confidence"] < CONF_THRESHOLD
        low_conf_indices = result_df[mask].index
        print(f"LLM processing {len(low_conf_indices)} low-confidence rows...")
        llm_result = run_llm_parallel(result_df, low_conf_indices, max_workers=6)
        for idx, category in llm_result.items():
            result_df.at[idx, "category"] = category

    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE}")
    print(result_df["category"].value_counts().to_string())
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label emails with rules (+ optional LLM)")
    parser.add_argument("--input", default=None, help="Input CSV path")
    parser.add_argument("--llm", action="store_true", help="Use local Ollama LLM for low-confidence rows")
    args = parser.parse_args()
    label_dataset(input_file=args.input, use_llm=args.llm)

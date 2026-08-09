import os
import pickle
import base64

from bs4 import BeautifulSoup
import re

import ftfy

import pandas as pd
from tqdm import tqdm

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request



SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_PATH = "credentials.json"
TOKEN_PATH = "token.pickle"
OUTPUT_CSV = "emails_dataset.csv"
OUTPUT_XLSX = "emails_dataset.xlsx"

def authenticate():
    creds = None

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            CREDENTIALS_PATH, SCOPES
        )
        creds = flow.run_local_server(port = 0)

        with open(TOKEN_PATH, "wb") as token:
            pickle.dump(creds, token)

    return build("gmail", "v1", credentials=creds)


def fetch_message_ids(service, limit=None):
    message_ids = []

    request = service.users().messages().list(
        userId = "me",
        maxResults = 500
    )

    while request:
        response = request.execute()
        messages = response.get("messages", [])

        for msg in messages:
            message_ids.append(msg["id"])

            if limit and len(message_ids) >= limit:
                return message_ids
            
        request = service.users().messages().list_next(request, response)

    return message_ids

def get_message(service, msg_id):
    return service.users().messages().get(
        userId="me",
        id=msg_id,
        format="full"
    ).execute()


def smart_decode(data):
    raw_bytes = base64.urlsafe_b64decode(data)
    try:
        return raw_bytes.decode("utf-8")
    except:
        try:
            return raw_bytes.decode("latin1").encode("utf-8").decode("utf-8")
        except:
            return raw_bytes.decode("utf-8", errors="ignore")

def decode_body(payload):
    mime_type = payload.get("mimeType", "")

    if "parts" not in payload:
        if mime_type in ("text/plain", "text/html"):
            data = payload.get("body", {}).get("data", "")
            if data:
                return smart_decode(data)
        return ""
    
    for part in payload.get("parts", []):
        text = decode_body(part)
        if text:
            return text
    
    return ""

def extract_features(message):
    headers = message["payload"]["headers"]

    subject, sender, recipient = None, None, None

    for h in headers:
        if h["name"] == "Subject":
            subject = h["value"]
        elif h["name"] == "From":
            sender = h["value"]
        elif h["name"] == "To":
            recipient = h["value"]
    
    raw_body = decode_body(message["payload"])
    body = clean_email_text(raw_body)
    snippet = message.get("snippet", "")
    labels = message.get("labelIds", [])

    text = (subject or "") + " " + (body or "")

    return {
        "subject": subject,
        "sender": sender,
        "recipient": recipient,
        "body": body,
        "snippet": snippet,
        "has_links": "http" in text.lower(),
        "labels": labels
    }


def clean_email_text(html):
    if not html:
        return ""
    
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.comment))):
        comment.extract()

    text = soup.get_text(separator=" ")

    text = ftfy.fix_text(text)
    text = re.sub(r'\b(\S{1,3})( \1){5,}\b', ' ', text)
    text = re.sub(r'(Í\s*)+', ' ', text)
    text = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text)
    text = re.sub(r'\b(\w+)( \1){3,}\b', r'\1', text)
    text = re.sub(r'http\S+|www\S+', ' <URL> ', text)

    patterns = [
        r"unsubscribe.*",
        r"terms of service.*",
        r"privacy policy.*",
        r"help center.*",
        r"account login.*"
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE)

    text = re.sub(r'-{3,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    text = ' '.join(text.split())

    return text.strip()


def save_batch(dataset, file_path):
    df = pd.DataFrame(dataset)

    df["body"] = df["body"].astype(str)
    df["body"] = df["body"].str.replace("\n", " ", regex=True)
    df["body"] = df["body"].str.replace('"', "'", regex=False)

    df.to_csv(
        file_path,
        mode="a",
        header=not os.path.exists(file_path),
        index=False,
        encoding="utf-8",
        quoting=1
    )

def run_collection(mode="batch", batch_size = 10000):
    service = authenticate()

    if mode == "test":
        limit = 5
    elif mode == "batch":
        limit = batch_size
    else:
        limit = None

    print(f"Fetching emails (mode = {mode})...")
    message_ids = fetch_message_ids(service, limit)
    print(f"Emails found: {len(message_ids)}")

    dataset = []

    for i, msg_id in enumerate(tqdm(message_ids)):
        try:
            msg = get_message(service, msg_id)
            features = extract_features(msg)

            dataset.append(features)

            if (i+1) % 1000 == 0:
                save_batch(dataset, OUTPUT_CSV)
                dataset = []

        except Exception as e:
            print(f"Error: {e}")
            continue

    if dataset:
        save_batch(dataset, OUTPUT_CSV)

    print("CSV saved successfully.")

    try:
        df = pd.read_csv(OUTPUT_CSV)
        df.to_excel(OUTPUT_XLSX, index=False)
        print("Excel file created successfully.")
    except Exception as e:
        print(f"Excel conversion failed: {e}")



if __name__ == "__main__":
    # run_collection(mode="test")
    # run_collection(mode="batch", batch_size=200)
    run_collection(mode="full")
# AI-Based Email Classifier

Personal project by [arshansari2880](https://github.com/arshansari2880).

Classifies emails into categories using Gmail data, rule-based + AI labeling, and an ensemble ML model (~94% accuracy).

## Features

- Collect emails via the Gmail API
- Clean and preprocess subject/body text
- Label with rules + AI-assisted labeling
- Train an ensemble classifier (Logistic Regression, Naive Bayes, Random Forest, SVM, XGBoost)
- Predict categories on new CSV data

## Project structure

| File | Purpose |
|------|---------|
| `email_collector.py` | Fetch emails from Gmail into a dataset |
| `labeling.py` | Clean text and assign labels |
| `model.py` | Train and save the classifier |
| `predict_csv.py` | Run predictions on a CSV |
| `count.py` | Quick label/count utilities |

## Setup

```bash
pip install -r requirements.txt
```

For Gmail collection, also install Google client libraries and place `credentials.json` from Google Cloud Console in this folder:

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## Usage

1. **Collect emails** (optional — needs Gmail OAuth):

   ```bash
   python email_collector.py
   ```

2. **Label the dataset:**

   ```bash
   python labeling.py
   ```

3. **Train the model:**

   ```bash
   python model.py
   ```

4. **Predict on a CSV:**

   ```bash
   python predict_csv.py
   ```

## Requirements

See `requirements.txt` (pandas, scikit-learn, xgboost, BeautifulSoup, etc.).

## License

MIT — feel free to use and adapt.

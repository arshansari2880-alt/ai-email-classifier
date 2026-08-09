# AI-Based Email Classifier

Personal project by [arshansari2880-alt](https://github.com/arshansari2880-alt).

Classifies emails into **spam / phishing / important / ham** using rule-based labeling and an ensemble ML model.

## Quick start (works offline)

```bash
pip install -r requirements.txt
python run_demo.py
```

This will:

1. Label `labeled_dataset.csv` with rules
2. Train and save the model
3. Predict categories for `input.csv` → `output_predictions.csv`

## Pipeline

| Step | Command | Input | Output |
|------|---------|-------|--------|
| Optional Gmail collect | `python email_collector.py --mode test` | `credentials.json` | `emails_dataset.csv` |
| Label | `python labeling.py` | `labeled_dataset.csv` or `emails_dataset.csv` | `labeled_fast.csv` |
| Train | `python model.py` | `labeled_fast.csv` | `email_classifier.pkl` |
| Predict | `python predict_csv.py` | `input.csv` | `output_predictions.csv` |
| Counts | `python count.py` | `labeled_fast.csv` | console summary |

Optional LLM refinement (needs [Ollama](https://ollama.com) + `mistral`):

```bash
python labeling.py --llm
```

## Gmail collection (optional)

1. Google Cloud → enable **Gmail API**
2. Create OAuth **Desktop** credentials
3. Save as `credentials.json` in this folder
4. Run:

```bash
python email_collector.py --mode test
```

Then rename/copy `emails_dataset.csv` usage is automatic — `labeling.py` picks it up if present (or keep using the included sample `labeled_dataset.csv`).

## Requirements

See `requirements.txt`.

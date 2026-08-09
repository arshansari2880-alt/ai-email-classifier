"""One-command demo: label sample emails -> train -> predict."""

from labeling import label_dataset
from model import main as train_main
from predict_csv import main as predict_main


def main():
    print("=== 1/3 Labeling ===")
    label_dataset(use_llm=False)

    print("\n=== 2/3 Training ===")
    train_main()

    print("\n=== 3/3 Predicting ===")
    predict_main()

    print("\nAll steps finished successfully.")


if __name__ == "__main__":
    main()

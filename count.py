import pandas as pd

df = pd.read_csv("labeled_fast.csv")
counts = df["category"].value_counts()
percent = df["category"].value_counts(normalize=True) * 100

summary = pd.DataFrame({
    "Count": counts,
    "Percentage": percent.round(2)
})

print(summary)
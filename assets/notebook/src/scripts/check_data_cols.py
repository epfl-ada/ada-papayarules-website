
import pandas as pd
try:
    df = pd.read_csv("data/soc-redditHyperlinks-body.tsv", sep="\t", nrows=5)
    print("Columns:", df.columns.tolist())
    # Check if any column looks like text
    for col in df.columns:
        print(f"Sample {col}: {df[col].iloc[0]}")
except Exception as e:
    print(e)

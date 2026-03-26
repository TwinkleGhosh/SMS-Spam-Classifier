import pandas as pd
import os

file_name = os.listdir("data")[0]
print("Using file:", file_name)

df = pd.read_csv(f"data/{file_name}", sep="\t", names=["label", "text"])
df.to_csv("data/spam.csv", index=False)

print("Converted successfully!")
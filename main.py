import pandas as pd
df = pd.read_csv("data/jigsaw/train.csv")
print("Shape:", df.shape)
print("First 30 columns:")
print(df.columns[:30])
print("\nSample rows:")
print(df[["comment_text","target"]].head(3))
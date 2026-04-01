import pandas as pd

df = pd.read_csv("data/housing.csv")
df = df.sample(frac=0.5, random_state=42)
df.to_csv("data/housing.csv", index=False)
exit()

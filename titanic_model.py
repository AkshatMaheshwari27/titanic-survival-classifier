import pandas as pd
import seaborn as sns

# 1. Load the dataset from seaborn
df = sns.load_dataset("titanic")

# 2. Inspect shape (rows, columns)
print("Dataset Shape:", df.shape)

# 3. Inspect data types and missing values
print("\n--- Data Info ---")
df.info()
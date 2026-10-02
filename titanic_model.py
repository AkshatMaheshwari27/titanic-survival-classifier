import pandas as pd
import seaborn as sns

# 1. Load the dataset from seaborn
df = sns.load_dataset("titanic")

# 2. Inspect shape (rows, columns)
print("Dataset Shape:", df.shape)

# 3. Inspect data types and missing values
print("\n--- Data Info ---")
df.info()

# --- CLEANING DATA ---

# 1. Fill missing 'age' values with the median (middle) age
df['age'] = df['age'].fillna(df['age'].median())

# 2. Convert text to numbers: 'female' -> 1, 'male' -> 0
df['sex'] = df['sex'].map({'female': 1, 'male': 0})

# 3. Select our features (X) and target (y)
features = ['pclass', 'sex', 'age', 'fare']
X = df[features]
y = df['survived']

# Verify the cleaned features have no missing values and are all numbers
print("\n--- Cleaned Features Info ---")
X.info()
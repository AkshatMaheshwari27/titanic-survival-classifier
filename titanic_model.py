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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --- TRAINING THE MODEL ---

# 1. Split the data (80% for studying, 20% for the final exam)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Create the blank baseline model
model = LogisticRegression()

# 3. FIT: Train the model using the training data and answers
model.fit(X_train, y_train)

# 4. PREDICT: Ask the model to take the exam on the hidden test features
predictions = model.predict(X_test)

# Print a sneak peek of the first 10 predictions vs the actual answers
print(f"\nFirst 10 predictions: {predictions[:10]}")
print(f"First 10 actuals:     {y_test.values[:10]}")

from sklearn.metrics import accuracy_score, confusion_matrix

# --- EVALUATING THE MODEL ---

# 1. Calculate percentage of correct guesses
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# 2. Generate the Confusion Matrix
print("\nConfusion Matrix:")
matrix = confusion_matrix(y_test, predictions)
print(matrix)

from sklearn.ensemble import RandomForestClassifier

# --- COMPARING A SECOND MODEL (RANDOM FOREST) ---

print("\n--- RANDOM FOREST MODEL ---")

# 1. Create the model (we use random_state so the trees grow the exact same way every time)
rf_model = RandomForestClassifier(random_state=42)

# 2. FIT (Study)
rf_model.fit(X_train, y_train)

# 3. PREDICT (Take the exam)
rf_predictions = rf_model.predict(X_test)

# 4. Evaluate
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
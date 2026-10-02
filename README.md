# Titanic Survival Predictor

A small machine learning pipeline that predicts passenger survival on the Titanic. It covers the full workflow: load, clean, split, train, evaluate, and compare.

## 📊 Dataset

- **Source:** Seaborn's built-in Titanic dataset (891 passengers).
- **Target:** `survived` (0 = no, 1 = yes).
- **Baseline features:** `pclass`, `sex`, `age`, `fare`.
- **Engineered features (added later):** `alone`, `who_man`, `who_woman`.

## 🛠️ Approach

1. **Load:** Import the dataset and inspect shape, data types, and missing values.
2. **Clean:** Fill missing `age` values with the median and map `sex` to binary (female = 1, male = 0).
3. **Engineer:** Convert `alone` to an integer and one-hot encode `who` (`drop_first=True` removes `who_child`, since a child is implied when `who_man` and `who_woman` are both 0).
4. **Split:** 80/20 train-test split with `random_state=42`.
5. **Train:** Fit Logistic Regression and Random Forest. Both use `random_state=42` so the comparison is fair.
6. **Evaluate:** Compare Accuracy and Confusion Matrices on the 179-passenger test set.

## 📈 Results

| Model                   | Baseline Accuracy | Baseline Matrix        | Engineered Accuracy | Engineered Matrix      |
| :---------------------- | :---------------- | :--------------------- | :------------------ | :--------------------- |
| **Logistic Regression** | 80.45%            | `[[90, 15], [20, 54]]` | 78.21%              | `[[88, 17], [22, 52]]` |
| **Random Forest**       | 79.33%            | `[[88, 17], [20, 54]]` | 81.01%              | `[[90, 15], [19, 55]]` |

_Matrix format: `[[TN, FP], [FN, TP]]`_

## 💡 Takeaway

With a test set of 179 passengers, one passenger is about 0.56% of the score. Engineered features lowered Logistic Regression by ~2.2% and raised Random Forest by ~1.7%. Both shifts amount to only 3-4 passengers, so they are within noise. Overall, the engineered features did not clearly help, and a simple model performed about as well as a more complex one.

## 🚀 How to Run

1. Clone this repository and open the folder.
2. Install the dependencies:

```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
```

3. Run the script:

```bash
   python titanic_model.py
```

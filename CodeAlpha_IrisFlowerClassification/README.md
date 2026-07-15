# CodeAlpha Data Science Internship — Task 1: Iris Flower Classification

## 📌 Objective
Build a machine learning model that classifies Iris flowers into one of three species
(*setosa*, *versicolor*, *virginica*) based on four measurements: sepal length, sepal
width, petal length, and petal width.

## 📊 Dataset
The classic Iris dataset (150 samples, 3 balanced classes), loaded directly via
`sklearn.datasets.load_iris()` — no external download required.

## 🛠 Approach
1. Load and explore the data (pairplots, correlation heatmap).
2. Split into train/test sets (80/20, stratified).
3. Scale features with `StandardScaler`.
4. Train and compare 4 classifiers:
   - Logistic Regression
   - K-Nearest Neighbors
   - Decision Tree
   - Random Forest
5. Evaluate with accuracy, precision/recall/F1, and a confusion matrix.
6. Select the best-performing model.

## ✅ Results
- Best model: **Logistic Regression**
- Test accuracy: **93.3%**
- All four models scored between 90–93% accuracy, showing the classes are
  well-separated by these measurements (especially *setosa*, which is
  perfectly separable).

## 📁 Files
- `iris_classification.py` — full pipeline (EDA → training → evaluation)
- `iris_pairplot.png` — pairwise feature relationships by species
- `iris_correlation_heatmap.png` — feature correlation matrix
- `iris_confusion_matrix.png` — confusion matrix for the best model
- `iris_model_comparison.png` — accuracy comparison across models

## ▶️ How to Run
```bash
pip install -r requirements.txt
python iris_classification.py
```

## 🧰 Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---
*Part of the CodeAlpha Data Science Internship.*

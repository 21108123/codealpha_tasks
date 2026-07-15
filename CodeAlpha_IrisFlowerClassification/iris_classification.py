"""
CodeAlpha Data Science Internship — Task 1
Iris Flower Classification

Trains and evaluates machine learning models to classify Iris flowers
(setosa, versicolor, virginica) based on their measurements.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("Dataset shape:", df.shape)
print(df.head())
print("\nClass distribution:\n", df["species"].value_counts())

# ---------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------
sns.pairplot(df, hue="species")
plt.savefig("iris_pairplot.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("iris_correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------
X = df.drop(columns="species")
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Train and compare multiple models
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
best_model_name, best_acc, best_model = None, 0, None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    if acc > best_acc:
        best_acc, best_model_name, best_model = acc, name, model

print("\n===================================")
print(f"Best model: {best_model_name} (Accuracy: {best_acc:.4f})")
print("===================================")

# ---------------------------------------------------------
# 5. Confusion matrix for the best model
# ---------------------------------------------------------
best_preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, best_preds, labels=iris.target_names)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_model_name}")
plt.savefig("iris_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 6. Model comparison chart
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.xticks(rotation=20)
plt.ylim(0, 1.05)
plt.savefig("iris_model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nAll plots saved: iris_pairplot.png, iris_correlation_heatmap.png, "
      "iris_confusion_matrix.png, iris_model_comparison.png")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Load & clean data
# ---------------------------------------------------------
df = pd.read_csv("Advertising.csv")
df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")

print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary stats:\n", df.describe())

# ---------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------
sns.pairplot(df, x_vars=["TV", "Radio", "Newspaper"], y_vars="Sales",
             height=4, kind="reg")
plt.savefig("sales_feature_relationships.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("sales_correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 3. Feature/target split
# ---------------------------------------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. Train models
# ---------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {}
best_name, best_r2, best_model = None, -np.inf, None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"\n--- {name} ---")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")

    if r2 > best_r2:
        best_r2, best_name, best_model = r2, name, model

print("\n===================================")
print(f"Best model: {best_name} (R2: {best_r2:.3f})")
print("===================================")

# ---------------------------------------------------------
# 5. Feature importance / coefficients for best model
# ---------------------------------------------------------
if best_name == "Linear Regression":
    importance = pd.Series(best_model.coef_, index=X.columns).sort_values(ascending=False)
    print("\nLinear Regression coefficients (impact per $1000 spend):")
else:
    importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature importances:")

print(importance)

plt.figure(figsize=(6, 4))
sns.barplot(x=importance.values, y=importance.index)
plt.title(f"Feature Impact on Sales — {best_name}")
plt.xlabel("Coefficient / Importance")
plt.savefig("sales_feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 6. Actual vs Predicted plot
# ---------------------------------------------------------
best_preds = best_model.predict(X_test)
plt.figure(figsize=(5, 5))
plt.scatter(y_test, best_preds, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales — {best_name}")
plt.savefig("sales_actual_vs_predicted.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nAll plots saved: sales_feature_relationships.png, sales_correlation_heatmap.png, "
      "sales_feature_importance.png, sales_actual_vs_predicted.png")

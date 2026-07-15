# CodeAlpha Data Science Internship — Task 4: Sales Prediction using Python

## 📌 Objective
Predict product sales based on advertising expenditure across three channels —
TV, Radio, and Newspaper — and analyze which channels drive the most impact,
to inform marketing strategy.

## 📊 Dataset
`Advertising.csv` — 200 records with columns `TV`, `Radio`, `Newspaper` (ad
spend in $ thousands) and `Sales` (in thousands of units).

> **Note:** This repo includes a sample dataset generated to match the
> structure of the standard advertising/sales dataset used for this task.
> If you download a specific dataset from the CodeAlpha task link, just
> replace `Advertising.csv` with it (same column names) — the script will
> work without any code changes.

## 🛠 Approach
1. Load and clean the data (check for nulls, drop stray index columns).
2. Explore relationships between each ad channel and Sales (scatter/reg plots,
   correlation heatmap).
3. Split into train/test sets (80/20).
4. Train and compare 2 regression models:
   - Linear Regression
   - Random Forest Regressor
5. Evaluate with MAE, RMSE, and R².
6. Interpret feature impact (coefficients / feature importances).
7. Visualize actual vs. predicted sales.

## ✅ Results
- Best model: **Linear Regression**
- R² score: **0.957** — the model explains ~96% of the variance in sales.
- **Radio** had the strongest per-dollar impact on sales, followed by TV,
  with Newspaper spend showing the weakest effect — suggesting marketing
  budget is best prioritized toward Radio and TV.

## 📁 Files
- `sales_prediction.py` — full pipeline (EDA → training → evaluation → insights)
- `Advertising.csv` — dataset
- `sales_feature_relationships.png` — spend vs. sales scatter/regression plots
- `sales_correlation_heatmap.png` — correlation matrix
- `sales_feature_importance.png` — channel impact on sales
- `sales_actual_vs_predicted.png` — model prediction accuracy plot

## ▶️ How to Run
```bash
pip install -r requirements.txt
python sales_prediction.py
```

## 🧰 Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---
*Part of the CodeAlpha Data Science Internship.*

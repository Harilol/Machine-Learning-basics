import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


# ── 1. Load Dataset ────────────────────────────────────────────────────────────

# sklearn's built-in diabetes dataset — 442 patients, 10 features
# Target: quantitative measure of disease progression one year after baseline
diabetes = load_diabetes()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

print("Dataset shape:", df.shape)
print(df.head())


# ── 2. Prepare Features & Labels ───────────────────────────────────────────────

X = df.drop(columns='target')   # 10 medical features
y = df['target']                 # Continuous disease progression score


# ── 3. Train / Test Split ──────────────────────────────────────────────────────

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ── 4. Hyperparameter Tuning with Grid Search ──────────────────────────────────

# Search over depth and leaf size combinations to find the best tree config
param_grid = {
    'max_depth':         [5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(),
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=5    # 5-fold cross-validation
)

grid_search.fit(X_train, y_train)

print("\nBest hyperparameters:", grid_search.best_params_)


# ── 5. Evaluate Best Model ─────────────────────────────────────────────────────

y_pred = grid_search.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error : {mae:.2f}")
print(f"R-squared           : {r2:.4f}")

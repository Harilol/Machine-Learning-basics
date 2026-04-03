# Diabetes Disease Progression Prediction

A machine learning project that predicts the **progression of diabetes** one year after baseline measurements, using a **Decision Tree Regressor** with hyperparameter tuning via Grid Search.

---

# Dataset

- **Source:** [`sklearn.datasets.load_diabetes`](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) — built into scikit-learn, no download needed
- **Origin:** Originally published by Bradley Efron et al. (Stanford), widely used as a regression benchmark
- **Size:** 442 patients, 10 features, 1 continuous target

> Note: All features in the dataset are already mean-centered and scaled by standard deviation, so no additional preprocessing is required.

# Features

| Feature | Description |
|---|---|
| age | Age (normalized) |
| sex | Sex (normalized) |
| bmi | Body Mass Index (normalized) |
| bp | Average blood pressure (normalized) |
| s1 | Total serum cholesterol (normalized) |
| s2 | Low-density lipoproteins (normalized) |
| s3 | High-density lipoproteins (normalized) |
| s4 | Total cholesterol / HDL ratio (normalized) |
| s5 | Log of serum triglycerides level (normalized) |
| s6 | Blood sugar level (normalized) |
| **target** | **Quantitative measure of disease progression after one year** |

---

# Problem Statement

Predicting how diabetes will progress in a patient is critical for personalized treatment planning. Rather than a simple yes/no diagnosis, this project tackles **regression** — estimating a continuous disease progression score — which gives clinicians a more nuanced view of a patient's trajectory.

---

# What We Did

1. **Loaded** sklearn's built-in diabetes dataset into a Pandas DataFrame
2. **Split** the data 80/20 into training and test sets
3. **Tuned hyperparameters** using `GridSearchCV` with 5-fold cross-validation across combinations of `max_depth`, `min_samples_split`, and `min_samples_leaf`
4. **Identified the best model** — `DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, min_samples_split=10)`
5. **Evaluated** performance on the held-out test set using MAE and R²

---

# Results

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | ~45.6 |
| R-squared (R²) | ~0.35 |

The R² of 0.35 means the model explains about 35% of the variance in disease progression. This is a reasonable baseline for a single Decision Tree on a noisy medical dataset — more complex models (Random Forest, Gradient Boosting) would likely improve this further.

---

# How to Run

### 1. Install dependencies

```bash
pip install scikit-learn pandas
```

### 2. Run the script

```bash
python diabetes_prediction.py
```

No dataset download needed — it loads automatically from scikit-learn.

---

# Tech Stack

- **Python 3.x**
- **Pandas** — data loading and exploration
- **Scikit-learn** — dataset, model, grid search, and evaluation metrics


---

# Possible Improvements

- Try **Random Forest** or **Gradient Boosting** for a stronger R² score
- Add **feature importance** visualization to see which medical factors matter most
- Use **cross-validated R²** as the primary metric for a more reliable estimate


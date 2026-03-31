# Telco Customer Churn Prediction 📉

Ever wondered why customers leave a telecom provider? That's exactly what this project tries to figure out — and more importantly, *predict* before it happens.

This is an end-to-end machine learning pipeline built on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It goes from raw CSV to a trained, saved model — covering data cleaning, exploratory analysis, and comparing three different classifiers to see which one does the best job.

---

## What's the problem?

Churn (customers cancelling their service) is one of the biggest headaches for telecom companies. Acquiring a new customer costs way more than keeping an existing one, so being able to flag "this customer is likely to leave" early is genuinely valuable. This project builds a binary classifier to do exactly that — predict whether a customer will churn or not based on their account details and usage patterns.

---

## What's in the dataset?

The dataset has ~7,000 customers with features like:

- **Demographics** — gender, senior citizen status, dependents
- **Account info** — tenure, contract type, billing method, monthly/total charges
- **Services** — phone, internet, streaming, tech support, etc.
- **Target** — `Churn` (Yes / No)

---

## Pipeline Overview

```
Load Data → Clean → EDA → Label Encode → SMOTE → Train → Evaluate → Save Model
```

**1. Data Cleaning**
The `TotalCharges` column had blank spaces instead of nulls for some rows. These get replaced with `0.0` and the column is cast to float.

**2. Exploratory Data Analysis (EDA)**
Before touching any model, the data gets explored visually:
- Histograms with mean/median overlaid for numerical columns
- Boxplots to spot outliers
- A correlation heatmap for the numerical features
- Count plots for every categorical column

**3. Label Encoding**
All text columns get converted to numbers using `LabelEncoder`. The encoders are saved in a dictionary in case you need to reverse the transformation later.

**4. Handling Class Imbalance (SMOTE)**
The dataset is imbalanced — most customers *don't* churn. If you train on this as-is, the model just learns to always predict "no churn" and still looks decent on accuracy. SMOTE fixes this by generating synthetic minority-class samples so both classes are equally represented during training.

**5. Model Training & Comparison**
Three models are compared using 5-fold cross-validation:
- Decision Tree
- Random Forest
- XGBoost

All three are then retrained on the full training set and evaluated on the held-out test set.

**6. Evaluation**
Each model is evaluated on:
- Accuracy score
- Confusion matrix
- Full classification report (precision, recall, F1)

**7. Saving the Model**
The Random Forest model gets saved as a `.pkl` file for future use or deployment.

---

## Results

| Model | Accuracy |
|---|---|
| Decision Tree | 72% |
| XGBoost | 83% |
| Random Forest | 84% |

> Random Forest came out on top, which is why it's saved as the final model.

---

## Project Structure

```
├── telco_churn_pipeline.py     # Main pipeline script
├── random_forest_model.pkl     # Saved model (generated after running)
├── Telco-Customer-Churn.csv    # Dataset (not included, link below)
└── README.md
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

**3. Add the dataset**
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project folder. Update the file path in the script if needed:
```python
df = pd.read_csv("Telco-Customer-Churn.csv")
```

**4. Run the script**
```bash
python telco_churn_pipeline.py
```

The script will print EDA output, cross-validation scores, and final evaluation metrics, then save the trained model as `random_forest_model.pkl`.

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing, models, evaluation |
| `imbalanced-learn` | SMOTE for class balancing |
| `xgboost` | XGBoost classifier |
| `pickle` | Saving the trained model |

---

## Things to Improve (Future Work)

- Hyperparameter tuning with GridSearchCV or Optuna
- Try OneHotEncoding instead of LabelEncoding for nominal columns
- Add a feature importance plot to understand what drives churn
- Build a simple prediction interface / API around the saved model

---

## Dataset Source

[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

Made as a learning project to practice a full ML pipeline from scratch. Feel free to fork, use, or suggest improvements!

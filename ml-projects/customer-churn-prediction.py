# Telco Customer Churn Prediction
# =============================================================================
# Predicts customer churn using Decision Tree, Random Forest, and XGBoost.
# Pipeline: Load → Clean → EDA → Encode → SMOTE → Train → Evaluate → Export
# ── Imports ───────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ── Load Data ─────────────────────────────────────────────────────────────────

df = pd.read_csv("/content/drive/MyDrive/Telco-Customer-Churn.csv")
df.head()

pd.set_option("display.max_columns", None)  # prevent column truncation in display

# customerID is just an identifier — not useful for prediction
df = df.drop(columns=["customerID"], axis=1)

df.shape
df.columns

# ── Data Understanding ────────────────────────────────────────────────────────

# Separate numerical columns so we can inspect only categorical ones below
numerical_data = ["tenure", "MonthlyCharges", "TotalCharges"]

# Print unique values for each categorical column to understand what we're working with
# (manual alternative: print(df["Gender"].unique()))
for i in df.columns:
    if i not in numerical_data:
        print(i, df[i].unique())
        print("-" * 50)

# ── Handle Missing Values ─────────────────────────────────────────────────────

# TotalCharges contains blank spaces instead of proper nulls — find and count them
df[df["TotalCharges"] == " "]
len(df[df["TotalCharges"] == " "])

# Replace blank spaces with "0.0" so the column can be cast to float
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
df["TotalCharges"] = df["TotalCharges"].astype(float)

# Check class balance in the target variable
df["Churn"].value_counts()

# ── EDA: Numerical Columns ────────────────────────────────────────────────────

def plot_hist(df, column_name):
    """Plot a histogram with KDE curve, mean, and median lines for a numerical column."""
    plt.figure(figsize=(5, 5))
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")

    # Calculate and overlay mean and median for a quick skewness check
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()

    plt.axvline(col_mean, linestyle="-", color="red", label="mean")
    plt.axvline(col_median, linestyle="--", color="green", label="median")
    plt.legend()
    plt.show()

plot_hist(df, "MonthlyCharges")
plot_hist(df, "tenure")
plot_hist(df, "TotalCharges")

def boxplot(df, column_name):
    """Plot a boxplot for a numerical column to visualise spread and outliers."""
    plt.figure(figsize=(5, 5))
    sns.boxplot(y=df[column_name])
    plt.title(f"Boxplot of {column_name}")
    plt.ylabel("boxplot")
    plt.show()

boxplot(df, "tenure")
boxplot(df, "TotalCharges")
boxplot(df, "MonthlyCharges")

# ── EDA: Correlation Heatmap ──────────────────────────────────────────────────

# Check how the three numerical features relate to each other
plt.figure(figsize=(6, 6))
sns.heatmap(
    df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(),
    annot=True,
    cmap="coolwarm",
)
plt.title("Correlation Heatmap")
plt.show()

# ── EDA: Categorical Columns ──────────────────────────────────────────────────

# Grab all object-type columns; also include SeniorCitizen (stored as int but is categorical)
object_cols = df.select_dtypes(include="object").columns.tolist()
object_cols = ["SeniorCitizen"] + object_cols

for i in object_cols:
    plt.figure(figsize=(5, 5))
    sns.countplot(x=df[i])
    plt.title(f"Count Plot of {i}")
    plt.xticks(rotation=45)
    plt.show()

# ── Label Encoding ────────────────────────────────────────────────────────────

# Convert categorical text columns to integers so models can process them.
# Note: LabelEncoder introduces an ordinal relationship which may not always
# be ideal — acceptable here since tree-based models are used throughout.

object_columns = df.select_dtypes(include="object").columns.tolist()

encoder = {}  # store encoders in case inverse_transform is needed later
for i in object_columns:
    label_encoder = LabelEncoder()
    df[i] = label_encoder.fit_transform(df[i])
    encoder[i] = label_encoder

df.head()

# ── Feature / Target Split ────────────────────────────────────────────────────

X = df.drop("Churn", axis=1)   # features
Y = df["Churn"]                 # target

# ── Train / Test Split ────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# ── Handle Class Imbalance with SMOTE ────────────────────────────────────────

# SMOTE generates synthetic samples for the minority class (churned customers)
# so the model doesn't become biased toward always predicting "no churn".
# Applied only on training data — the test set is left untouched.
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

X_train_smote.shape, y_train_smote.shape
X_test.shape, y_test.shape

print("Class distribution after SMOTE:")
print(y_train_smote.value_counts())

# ── Model Training with Cross-Validation ─────────────────────────────────────

# Compare three classifiers using 5-fold cross-validation on the balanced data
models = {
    "decision tree": DecisionTreeClassifier(random_state=42),
    "random forest": RandomForestClassifier(random_state=42),
    "xgboost":       XGBClassifier(random_state=42),
}

scores = {}
for model_name, model in models.items():
    print(f"Training {model_name} with default parameters")
    score = cross_val_score(
        model, X_train_smote, y_train_smote, cv=5, scoring="accuracy"
    )
    scores[model_name] = score
    print(f"{model_name} cross-validation accuracy: {np.mean(score):.2f}")
    print("-" * 50)

# ── Final Model Training ──────────────────────────────────────────────────────

# Re-train each model on the full SMOTE training set for final evaluation
rfc = RandomForestClassifier()
rfc.fit(X_train_smote, y_train_smote)

xgb = XGBClassifier()
xgb.fit(X_train_smote, y_train_smote)

dec = DecisionTreeClassifier()
dec.fit(X_train_smote, y_train_smote)

# ── Predictions ───────────────────────────────────────────────────────────────

rfc_pred = rfc.predict(X_test)
xgb_pred = xgb.predict(X_test)
dec_pred = dec.predict(X_test)

# ── Evaluation ────────────────────────────────────────────────────────────────

# Accuracy — overall percentage of correct predictions
print("Accuracy score of Random Forest :", accuracy_score(y_test, rfc_pred))
print("Accuracy score of XGBoost       :", accuracy_score(y_test, xgb_pred))
print("Accuracy score of Decision Tree :", accuracy_score(y_test, dec_pred))

# Confusion matrix — shows true/false positives and negatives per class
print("\nConfusion matrix — Random Forest:\n", confusion_matrix(y_test, rfc_pred))
print("\nConfusion matrix — XGBoost:\n",       confusion_matrix(y_test, xgb_pred))
print("\nConfusion matrix — Decision Tree:\n",  confusion_matrix(y_test, dec_pred))

# Classification report — precision, recall, and F1-score per class
print("\nClassification report — Random Forest:\n", classification_report(y_test, rfc_pred))
print("\nClassification report — XGBoost:\n",       classification_report(y_test, xgb_pred))
print("\nClassification report — Decision Tree:\n",  classification_report(y_test, dec_pred))

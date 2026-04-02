# ==========================================
# 0) ENVIRONMENT / PRE-SETUP (Imports + Settings)
# ==========================================

# Core
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn: split, pipeline, preprocessing, model, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)

# Plot settings (beginner friendly)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
RANDOM_STATE = 42


# ==========================================
# 1) LOAD DATA (Google Colab upload)
# ==========================================
from google.colab import files
uploaded = files.upload()  # choose creditcard.csv

df = pd.read_csv("creditcard.csv")

print("Shape:", df.shape)
display(df.head())

print("\nColumns:", df.columns.tolist())
print("\nMissing values (top 10):")
display(df.isna().sum().sort_values(ascending=False).head(10))


# ==========================================
# 2) QUICK DATA CHECKS
# ==========================================
# Expect columns: Time, V1..V28, Amount, Class
target_col = "Class"
assert target_col in df.columns, "Class column not found!"

print("\nClass counts:")
display(df[target_col].value_counts())

print("\nClass rates:")
display(df[target_col].value_counts(normalize=True))


# ==========================================
# 3) VISUALIZATION / EDA 
# ==========================================

# --- 3.1 Class imbalance plot ---
plt.figure(figsize=(10,4))
ax = sns.countplot(data=df, x="Class")
ax.set_title("Class distribution (0=Normal, 1=Fraud)")
ax.bar_label(ax.containers[0])
plt.show()

fraud_rate = df["Class"].mean()
print(f"Fraud rate: {fraud_rate:.4%}")


# --- 3.2 Amount distribution (all) ---
plt.figure(figsize=(10,4))
sns.histplot(data=df, x="Amount", bins=50, kde=True)
plt.title("Amount distribution (all transactions)")
plt.show()


# --- 3.3 Amount distribution by Class (density) ---
plt.figure(figsize=(10,4))
sns.histplot(
    data=df, x="Amount", hue="Class",
    bins=60, element="step",
    stat="density", common_norm=False
)
plt.title("Amount distribution by Class (density)")
plt.show()


# --- 3.4 Amount distribution by Class (log-scale on x) ---
# (Helps because Amount can have a long tail)
plt.figure(figsize=(10,4))
sns.histplot(
    data=df, x="Amount", hue="Class",
    bins=60, element="step",
    stat="density", common_norm=False,
    log_scale=(True, False)
)
plt.title("Amount distribution by Class (log-scaled x)")
plt.show()


# --- 3.5 Time distribution ---
plt.figure(figsize=(10,4))
sns.histplot(data=df, x="Time", bins=60)
plt.title("Time distribution (seconds since first transaction)")
plt.show()


# --- 3.6 Time mapped to hour-of-day (approx: Time % 24h) ---
df_tmp = df.copy()
df_tmp["Hour"] = (df_tmp["Time"] / 3600.0) % 24

plt.figure(figsize=(10,4))
sns.histplot(data=df_tmp, x="Hour", bins=24)
plt.title("Transactions by hour of day (approx; Time % 24h)")
plt.show()


# --- 3.7 Correlation heatmap (top correlated features with Class) ---
corr = df.corr(numeric_only=True)
top_features = corr["Class"].abs().sort_values(ascending=False).head(12).index

plt.figure(figsize=(8,6))
sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation heatmap (top features by |corr| with Class)")
plt.show()


# ==========================================
# 4) TRAIN/TEST SPLIT (Stratified)
# ==========================================
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train fraud rate:", y_train.mean(), "Test fraud rate:", y_test.mean())


# ==========================================
# 5) ML BASELINE MODEL (Logistic Regression + Scaling)
# ==========================================
# We scale all numeric columns (Time, V1..V28, Amount).
numeric_cols = X_train.columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder="drop"
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

clf.fit(X_train, y_train)


# ==========================================
# 6) EVALUATION (Metrics + Beginner-friendly Visuals)
# ==========================================

# --- 6.1 Predict probabilities (important for PR curve + threshold tuning) ---
y_proba = clf.predict_proba(X_test)[:, 1]

# Default threshold = 0.5
threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)

print("\n==== Metrics (threshold=0.5) ====")
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("PR-AUC (Average Precision):", average_precision_score(y_test, y_proba))
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))


# --- 6.2 Confusion matrix (heatmap) ---
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (threshold=0.5)")
plt.show()


# --- 6.3 Precision-Recall curve (best for imbalanced datasets) ---
prec, rec, pr_thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(6,5))
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.grid(True)
plt.show()


# --- 6.4 ROC curve (common, but PR is more meaningful here) ---
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="Model")
plt.plot([0,1], [0,1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# 7) THRESHOLD TUNING (Visualize Precision/Recall vs Threshold)
# ==========================================
thresholds = np.linspace(0.01, 0.99, 99)

precision_list = []
recall_list = []

for t in thresholds:
    pred_t = (y_proba >= t).astype(int)

    tp = ((pred_t == 1) & (y_test == 1)).sum()
    fp = ((pred_t == 1) & (y_test == 0)).sum()
    fn = ((pred_t == 0) & (y_test == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    precision_list.append(precision)
    recall_list.append(recall)

plt.figure(figsize=(10,4))
plt.plot(thresholds, precision_list, label="Precision")
plt.plot(thresholds, recall_list, label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision/Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()

# Example: pick a threshold targeting high recall (e.g., >= 0.90)
target_recall = 0.90
best_t = None
best_f1 = -1

for t, p, r in zip(thresholds, precision_list, recall_list):
    f1 = (2*p*r)/(p+r) if (p+r) else 0
    if r >= target_recall and f1 > best_f1:
        best_f1 = f1
        best_t = t

print("Suggested threshold for recall >= 0.90:", best_t, "with best F1 among those:", best_f1)

if best_t is not None:
    y_pred_best = (y_proba >= best_t).astype(int)
    cm_best = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm_best, annot=True, fmt="d", cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (threshold={best_t:.2f})")
    plt.show()

    print("\n==== Metrics (tuned threshold) ====")
    print(classification_report(y_test, y_pred_best, digits=4))
else:
    print("No threshold achieved the target recall; try lowering target_recall or improving the model.")
"""
Task 2: Predictive Modeling Using Machine Learning
===================================================
Demonstrates Linear Regression, Decision Tree, and Random Forest models.
Trains and evaluates each model, then visualizes performance with
confusion matrices and ROC curves.
 
Run:
    pip install scikit-learn matplotlib seaborn pandas numpy
    python predictive_modeling.py
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression   # classification analogue of linear regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
 
# ── 1. Load dataset ──────────────────────────────────────────────────────────
# Using the Breast Cancer dataset (binary classification, built into sklearn)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign
 
print("=" * 60)
print("Dataset: Breast Cancer (sklearn built-in)")
print(f"  Samples  : {X.shape[0]}")
print(f"  Features : {X.shape[1]}")
print(f"  Classes  : {list(data.target_names)}")
print("=" * 60)
 
# ── 2. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# ── 3. Feature scaling (important for Logistic Regression) ──────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
 
# ── 4. Define models ─────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=10_000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}
 
results = {}
 
for name, model in models.items():
    # Logistic Regression uses scaled features; tree models use raw
    X_tr = X_train_sc if name == "Logistic Regression" else X_train
    X_te = X_test_sc  if name == "Logistic Regression" else X_test
 
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]   # probability for ROC curve
 
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
 
    results[name] = {
        "model":   model,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
        "acc":     acc,
        "cm":      cm,
        "fpr":     fpr,
        "tpr":     tpr,
        "auc":     roc_auc,
    }
 
    print(f"\n── {name} ──")
    print(f"  Accuracy : {acc:.4f}  |  AUC : {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
 
# ── 5. Visualise ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Task 2 – Predictive Modeling Using Machine Learning",
             fontsize=15, fontweight="bold", y=1.01)
 
colors = ["#4C72B0", "#55A868", "#C44E52"]
 
# Row 1 – Confusion matrices
for ax, (name, res), color in zip(axes[0], results.items(), colors):
    sns.heatmap(
        res["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=data.target_names,
        yticklabels=data.target_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{name}\nAccuracy: {res['acc']:.2%}", fontsize=11)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
 
# Row 2 – ROC curves (one per model)
for ax, (name, res), color in zip(axes[1], results.items(), colors):
    ax.plot(res["fpr"], res["tpr"], color=color, lw=2,
            label=f"AUC = {res['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {name}", fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
 
plt.tight_layout()
plt.savefig("model_performance.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved as model_performance.png")
 
# ── 6. Feature importance (Random Forest) ────────────────────────────────────
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top10 = importances.nlargest(10)
 
fig2, ax2 = plt.subplots(figsize=(9, 5))
top10.sort_values().plot(kind="barh", ax=ax2, color="#55A868", edgecolor="white")
ax2.set_title("Top 10 Feature Importances – Random Forest", fontsize=12, fontweight="bold")
ax2.set_xlabel("Importance Score")
ax2.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Feature importance plot saved as feature_importance.png")
 
# ── 7. Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Model':<25} {'Accuracy':>10} {'AUC':>10}")
print("-" * 45)
for name, res in results.items():
    print(f"{name:<25} {res['acc']:>10.4f} {res['auc']:>10.4f}")
print("=" * 60)
 
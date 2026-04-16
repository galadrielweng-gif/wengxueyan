import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("../data/heart_failure_clinical_records_dataset.csv")

X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# =========================
# 2. train/test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 3. SMOTE（🔥冲刺关键）
# =========================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# =========================
# 4. 标准化（LR用）
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5. Logistic Regression
# =========================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train_sm)

lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

# =========================
# 6. Random Forest
# =========================
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train_sm, y_train_sm)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# =========================
# 7. XGBoost（🔥加分模型）
# =========================
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric="logloss"
)

xgb.fit(X_train_sm, y_train_sm)

xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]

# =========================
# 8. 评估函数
# =========================
def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))
    print("AUC:", roc_auc_score(y_true, y_prob))

# =========================
# 9. 输出结果
# =========================
evaluate("Logistic Regression (SMOTE)", y_test, lr_pred, lr_prob)
evaluate("Random Forest (SMOTE)", y_test, rf_pred, rf_prob)
evaluate("XGBoost (FINAL)", y_test, xgb_pred, xgb_prob)

# =========================
# 10. ROC曲线
# =========================
plt.figure()

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)

plt.plot(fpr_lr, tpr_lr, label="LR")
plt.plot(fpr_rf, tpr_rf, label="RF")
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost")

plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve (Final Models)")

plt.savefig("../figures/roc_final.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 11. Feature Importance（XGBoost）
# =========================
importances = xgb.feature_importances_
features = X.columns

feat_imp = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n===== TOP RISK FACTORS (XGBoost) =====")
print(feat_imp.head(5))

plt.figure()
plt.barh(feat_imp["feature"], feat_imp["importance"])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance")

plt.savefig("../figures/xgb_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

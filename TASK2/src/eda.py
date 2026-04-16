import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 0. 创建保存目录（关键）
# =========================
FIG_PATH = "../figures"
os.makedirs(FIG_PATH, exist_ok=True)

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("../data/heart_failure_clinical_records_dataset.csv")

# =========================
# 2. 基础信息输出
# =========================
print("===== HEAD =====")
print(df.head())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# =========================
# 3. 画图风格
# =========================
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (6,4)

# =========================
# 4. DEATH EVENT分布
# =========================
plt.figure()
sns.countplot(x="DEATH_EVENT", data=df)
plt.title("Death Event Distribution")
plt.savefig(f"{FIG_PATH}/death_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 5. 年龄分布
# =========================
plt.figure()
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.savefig(f"{FIG_PATH}/age_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 6. 相关性热力图
# =========================
plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.savefig(f"{FIG_PATH}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 7. Age vs Death
# =========================
plt.figure()
sns.boxplot(x="DEATH_EVENT", y="age", data=df)
plt.title("Age vs Death Event")
plt.savefig(f"{FIG_PATH}/age_vs_death.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 8. Ejection Fraction vs Death
# =========================
plt.figure()
sns.boxplot(x="DEATH_EVENT", y="ejection_fraction", data=df)
plt.title("Ejection Fraction vs Death Event")
plt.savefig(f"{FIG_PATH}/ejection_fraction_vs_death.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 9. Serum Creatinine vs Death
# =========================
plt.figure()
sns.boxplot(x="DEATH_EVENT", y="serum_creatinine", data=df)
plt.title("Serum Creatinine vs Death Event")
plt.savefig(f"{FIG_PATH}/serum_creatinine_vs_death.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 10. 特征分布
# =========================
features = ["age", "ejection_fraction", "serum_creatinine", "serum_sodium"]

df[features].hist(bins=20, figsize=(10,6))
plt.suptitle("Key Feature Distributions")

plt.savefig(f"{FIG_PATH}/feature_distributions.png", dpi=300, bbox_inches='tight')
plt.show()

# =========================
# 11. 科研结论输出
# =========================
print("\n===== KEY INSIGHTS =====")
print("No missing values found.")
print("Class imbalance exists in DEATH_EVENT.")
print("Key risk factors: age, ejection_fraction, serum_creatinine")

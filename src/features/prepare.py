# __define-ocg__: Data preparation for Credit Card Default Prediction (MLOps)
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# -------------------------
# 🔧 Define file paths
# -------------------------
RAW_PATH = Path("src/data/raw/default of credit card clients.xls")
PROC_PATH = Path("src/data/processed")
REF_PATH = Path("src/data/reference")

PROC_PATH.mkdir(parents=True, exist_ok=True)
REF_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------
# 🧠 Load dataset
# -------------------------
print("📂 Loading dataset...")
df = pd.read_excel(RAW_PATH, header=1)
df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
df.rename(columns={"default_payment_next_month": "target"}, inplace=True)

# Fix column naming issue: rename pay_0 → pay_1 if needed
if "pay_0" in df.columns:
    df.rename(columns={"pay_0": "pay_1"}, inplace=True)

print(f"✅ Loaded dataset with shape: {df.shape}")
print(f"Columns: {list(df.columns[:10])} ...")

# -------------------------
# ⚙️ Feature Engineering
# -------------------------
print("🧮 Engineering features...")

# Utilization ratios and payment ratios
for i in range(1, 7):
    df[f"util_{i}"] = df[f"bill_amt{i}"] / df["limit_bal"].replace(0, 1)
    df[f"pay_ratio_{i}"] = df[f"pay_amt{i}"] / df[f"bill_amt{i}"].replace(0, 1)

# Average utilization across 6 months
df["avg_util"] = df[[f"util_{i}" for i in range(1, 7)]].mean(axis=1)

# Missed payment count (how many months had delayed payments)
df["misspay_cnt"] = (df[[f"pay_{i}" for i in range(1, 7)]] > 0).sum(axis=1)

# -------------------------
# 🎯 Split data into train/test
# -------------------------
print("📊 Splitting data into train/test...")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train = X_train.copy()
train["target"] = y_train

test = X_test.copy()
test["target"] = y_test

# -------------------------
# 💾 Save processed datasets
# -------------------------
train.to_csv(PROC_PATH / "train.csv", index=False)
test.to_csv(PROC_PATH / "test.csv", index=False)
train.sample(min(10000, len(train)), random_state=7).drop(columns=["target"]).to_csv(
    REF_PATH / "reference.csv", index=False
)

print("\n✅ Data preparation complete!")
print(" - src/data/processed/train.csv")
print(" - src/data/processed/test.csv")
print(" - src/data/reference/reference.csv")

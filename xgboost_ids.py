import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

# ==========================================
# 1. LOAD DATA
# ==========================================

train_files = train_files = [ "data/Monday-WorkingHours.pcap_ISCX.csv",
                              "data/Tuesday-WorkingHours.pcap_ISCX.csv",
                              "data/Wednesday-workingHours.pcap_ISCX.csv",
                              "data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                              "data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                              "data/Friday-WorkingHours-Morning.pcap_ISCX.csv",
                              "data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                              "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" ]

dfs = []

for f in train_files:
    dfs.append(pd.read_csv(f, low_memory=False))

df = pd.concat(dfs, ignore_index=True)

print("Dataset shape:", df.shape)

# ==========================================
# 2. CLEAN DATA
# ==========================================

df.columns = df.columns.str.strip()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# ==========================================
# 3. BALANCE DATA (Random UnderSampling)
# ==========================================

benign_df = df[df["Label"] == "BENIGN"]
attack_df = df[df["Label"] != "BENIGN"]

print("\nBefore balancing")
print("-----------------------")
print("Benign :", len(benign_df))
print("Attack :", len(attack_df))

# Lấy ngẫu nhiên số mẫu Benign bằng số mẫu Attack
benign_sample = benign_df.sample(
    n=len(attack_df),
    random_state=42
)

# Ghép lại
df = pd.concat([benign_sample, attack_df], ignore_index=True)

# Trộn dữ liệu
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter balancing")
print("-----------------------")
print(df["Label"].value_counts())

# ==========================================
# 4. LABEL
# ==========================================

y = np.where(df["Label"] == "BENIGN", 0, 1)

# ==========================================
# 5. FEATURES
# ==========================================

X = df.select_dtypes(include=np.number)

print("\nNumber of features:", X.shape[1])

# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ==========================================
# 7. STANDARD SCALER
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 8. XGBOOST
# ==========================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

print("Training completed.")

# ==========================================
# 9. PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 10. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("\n==============================")
print("      EVALUATION RESULT")
print("==============================")

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

print("\nConfusion Matrix")
print(cm)

print("\nDetailed Results")
print("------------------------------")
print(f"TP : {TP}")
print(f"FP : {FP}")
print(f"FN : {FN}")
print(f"TN : {TN}")

false_alarm_rate = FP / (FP + TN) * 100
miss_rate = FN / (FN + TP) * 100

print("\nRates")
print("------------------------------")
print(f"False Alarm Rate  : {false_alarm_rate:.2f}%")
print(f"Missed Attack Rate: {miss_rate:.2f}%")
print(f"Detection Rate    : {recall*100:.2f}%")

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=["BENIGN", "ATTACK"]
))
import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

TRAIN_FPATH = os.path.join(DATA_DIR, "train.csv")

df_train = pd.read_csv(TRAIN_FPATH)
df_patients = df_train.groupby("patient_id").agg({"target": lambda series: series.max()}).reset_index()

skf = StratifiedKFold()
for i, (train, valid) in enumerate(skf.split(df_patients.patient_id, df_patients.target)):
    print(f"Fold {i + 1}:")

    fold_dir = os.path.join(DATA_DIR, f"fold_{i + 1}")
    if not os.path.isdir(fold_dir):
        os.makedirs(fold_dir)

    train_patients = df_patients.iloc[train].patient_id
    df_fold_train = df_train[df_train.patient_id.isin(train_patients)]
    train_counts = train_fold.target.value_counts()

    neg = train_counts[0]
    pos = train_counts[1]
    print(f"train: {pos / (pos + neg)}")

    train_fold.to_csv(os.path.join(fold_dir, f"train.csv"), index=False)

    valid_patients = df_patients.iloc[valid].patient_id
    df_fold_valid = df_train[df_train.patient_id.isin(valid_patients)]
    valid_counts = valid_fold.target.value_counts()

    neg = valid_counts[0]
    pos = valid_counts[1]
    print(f"valid: {pos / (pos + neg)}")

    valid_fold.to_csv(os.path.join(fold_dir, f"valid.csv"), index=False)

    print("-------")

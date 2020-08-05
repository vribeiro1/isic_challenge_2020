import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FPATH = os.path.join(DATA_DIR, "train.csv")

df = pd.read_csv(TRAIN_FPATH)

skf = StratifiedKFold()
for i, (train, valid) in enumerate(skf.split(df, df.target)):
    print(f"Fold {i + 1}:")

    fold_dir = os.path.join(DATA_DIR, f"fold_{i + 1}")
    if not os.path.isdir(fold_dir):
        os.makedirs(fold_dir)

    df_train = df.iloc[train]
    train_counts = df_train.target.value_counts()

    neg = train_counts[0]
    pos = train_counts[1]
    print(f"train: {pos / (pos + neg)} ({pos}/{(pos + neg)})")

    df_train.to_csv(os.path.join(fold_dir, f"train.csv"), index=False)

    df_valid = df.iloc[valid]
    valid_counts = df_valid.target.value_counts()

    neg = valid_counts[0]
    pos = valid_counts[1]
    print(f"valid: {pos / (pos + neg)} ({pos}/{pos + neg})")

    df_valid.to_csv(os.path.join(fold_dir, f"valid.csv"), index=False)

    print("-------")

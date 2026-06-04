import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold


def load_config(config_path="./config.yaml"):
    """Loads the master configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_skab(config):
    """
    Loads SKAB files, assigns group IDs for Cross-Validation,
    adds source_file/source_group metadata, and cleans targets.
    """
    skab_config = config["data"]["skab"]
    base_dir = skab_config["dir_path"]
    folders = skab_config["target_folders"]

    df_list = []
    group_id = 0

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)

                df = pd.read_csv(
                    file_path,
                    sep=";",
                    index_col="datetime",
                    parse_dates=True
                )

                if "changepoint" in df.columns:
                    df = df.drop(columns=["changepoint"])

                df["source_group"] = group_id
                df["source_file"] = file

                df_list.append(df)
                group_id += 1

    if not df_list:
        raise FileNotFoundError(
            "No SKAB CSV files were found. Check data.skab.dir_path and target_folders."
        )

    return pd.concat(df_list)


def _add_pc1_features(X_train_scaled, X_test_scaled, X_val_scaled=None):
    """
    Fits PCA only on train data and returns PC1 arrays for automata.

    This avoids data leakage:
    - PCA is fit on train only
    - validation/test are transformed with the same PCA object
    """
    pca = PCA(n_components=1)

    X_train_pc1 = pca.fit_transform(X_train_scaled).reshape(-1)
    X_test_pc1 = pca.transform(X_test_scaled).reshape(-1)

    if X_val_scaled is None:
        return X_train_pc1, X_test_pc1

    X_val_pc1 = pca.transform(X_val_scaled).reshape(-1)
    return X_train_pc1, X_val_pc1, X_test_pc1


def prepare_skab_cv(df, config):
    """Yields scaled and PCA-transformed Train/Test splits using GroupKFold."""
    n_splits = config["data"]["skab"]["n_splits_cv"]
    gkf = GroupKFold(n_splits=n_splits)

    X = df.drop(columns=["anomaly", "source_group", "source_file"])
    y = df["anomaly"].values
    groups = df["source_group"].values

    splits = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_pc1, X_test_pc1 = _add_pc1_features(
            X_train_scaled,
            X_test_scaled
        )

        splits.append({
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "X_train_pc1": X_train_pc1,
            "X_test_pc1": X_test_pc1
        })

    return splits


def load_and_prepare_batadal(config):
    """Loads BATADAL, cleans labels, and performs chronological 60/20/20 split."""
    batadal_config = config["data"]["batadal"]
    file_path = os.path.join(
        batadal_config["dir_path"],
        batadal_config["main_file"]
    )

    df = pd.read_csv(
        file_path,
        sep=",",
        skipinitialspace=True,
        index_col="DATETIME",
        parse_dates=True
    )

    df["ATT_FLAG"] = df["ATT_FLAG"].apply(lambda x: 0 if x == -999 else 1)

    X = df.drop(columns=["ATT_FLAG"])
    y = df["ATT_FLAG"].values

    n = len(df)
    ratios = batadal_config["split_ratios"]

    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    X_train, y_train = X.iloc[:train_end], y[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y[val_end:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_pc1, X_val_pc1, X_test_pc1 = _add_pc1_features(
        X_train_scaled,
        X_test_scaled,
        X_val_scaled
    )

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "X_train_pc1": X_train_pc1,
        "X_val_pc1": X_val_pc1,
        "X_test_pc1": X_test_pc1
    }


if __name__ == "__main__":
    cfg = load_config()

    print("Loading BATADAL...")
    batadal_data = load_and_prepare_batadal(cfg)
    print(f"BATADAL Train Shape: {batadal_data['X_train'].shape}")
    print(f"BATADAL Train PC1 Shape: {batadal_data['X_train_pc1'].shape}")

    print("Loading SKAB...")
    skab_df = load_skab(cfg)
    skab_splits = prepare_skab_cv(skab_df, cfg)
    print(f"SKAB CV Fold 1 Train Shape: {skab_splits[0]['X_train'].shape}")
    print(f"SKAB CV Fold 1 Train PC1 Shape: {skab_splits[0]['X_train_pc1'].shape}")
import numpy as np
import torch
from load_data import load_raw

CLASS_LETTERS = ["N", "S", "V", "F", "Q"]
EXPECTED_TRAIN = {0: 72471, 1: 2223, 2: 5788, 3: 641, 4: 6431}
EXPECTED_TEST  = {0: 18118, 1: 556, 2: 1448, 3: 162, 4: 1608}


def class_counts(y, expected, name):
    """Print rows per class and compare with the table in Section 1."""
    counts = np.bincount(y, minlength=5)
    print(f"\nRows per class, {name} file:")
    print(f"{'class':<6}{'found':>10}{'expected':>10}{'match':>8}")
    for k in range(5):
        ok = counts[k] == expected[k]
        print(f"{CLASS_LETTERS[k]:<6}{counts[k]:>10}{expected[k]:>10}{str(ok):>8}")
    print(f"total found = {counts.sum()}, expected = {sum(expected.values())}")
    return counts


def prepare(data_dir="data", verbose=True):
    # ---- 1.1 Load and print shapes -------------------------------------
    raw = load_raw(data_dir)
    train_df, test_df = raw["train"], raw["test"]
    if verbose:
        print("1.1 Shapes")
        print("  train:", train_df.shape)
        print("  test :", test_df.shape)

    # ---- 1.2 Separate signal (cols 0-186) from label (col 187) ---------
    X_train_all = train_df.iloc[:, :187].to_numpy()
    X_test      = test_df.iloc[:, :187].to_numpy()
    y_train_raw = train_df.iloc[:, 187]
    y_test_raw  = test_df.iloc[:, 187]

    if verbose:
        print("\n1.2 Label column")
        print("  dtype before conversion:", y_train_raw.dtype)
        print("  unique values (train):", sorted(y_train_raw.unique()))
        print("  all whole numbers:", bool(np.all(y_train_raw.to_numpy() % 1 == 0)))

    y_train_all = y_train_raw.to_numpy().astype(np.int64)
    y_test      = y_test_raw.to_numpy().astype(np.int64)

    if verbose:
        print("  dtype after conversion:", y_train_all.dtype)
        print("  signal columns:", X_train_all.shape[1], "(must be 187)")

    # ---- 1.3 Min / max signal value (training file) --------------------
    if verbose:
        print("\n1.3 Signal range (training file)")
        print("  min =", X_train_all.min())
        print("  max =", X_train_all.max())

    # ---- 1.4 Rows per class, both files --------------------------------
    if verbose:
        class_counts(y_train_all, EXPECTED_TRAIN, "train")
        class_counts(y_test, EXPECTED_TEST, "test")

    # ---- 1.6 Train / validation split (exact code from the assignment) -
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X_train_all[train_idx], y_train_all[train_idx]
    X_val,   y_val   = X_train_all[val_idx],   y_train_all[val_idx]

    if verbose:
        print("\n1.6 Split sizes")
        print("  train     :", len(X_train))
        print("  validation:", len(X_val))
        print("  test      :", len(X_test))

    # ---- 1.7 Tensors ---------------------------------------------------
    data = {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "X_val":   torch.tensor(X_val,   dtype=torch.float32),
        "y_val":   torch.tensor(y_val,   dtype=torch.long),
        "X_test":  torch.tensor(X_test,  dtype=torch.float32),
        "y_test":  torch.tensor(y_test,  dtype=torch.long),
    }

    if verbose:
        print("\n1.7 Tensors")
        for k, v in data.items():
            print(f"  {k:<8} shape={tuple(v.shape)}  dtype={v.dtype}")

    return data


if __name__ == "__main__":
    prepare()

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare import prepare
from model import ECGNet
from metrics import accuracy, confusion_matrix, per_class_accuracy, balanced_accuracy

LETTERS = ["N", "S", "V", "F", "Q"]
NAMES = ["N: normal", "S: supraventricular", "V: ventricular", "F: fusion", "Q: unknown/paced"]


def main():
    data = prepare(verbose=False)
    X_test, y_test = data["X_test"], data["y_test"]

    # 5.1 Load the saved parameters and predict: class with the largest logit
    net = ECGNet()
    net.load_state_dict(torch.load("results/best_model.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        logits = net(X_test)
    y_pred = logits.argmax(dim=1).numpy()
    y_true = y_test.numpy()

    # 5.2 / 5.3 Metrics (all written by hand in metrics.py)
    acc = accuracy(y_true, y_pred)
    C = confusion_matrix(y_true, y_pred)
    pca = per_class_accuracy(C)
    bal = balanced_accuracy(C)

    # Task 6.1: a model that always answers "N"
    counts = np.bincount(y_true, minlength=5)
    always_n = counts[0] / counts.sum()

    # Correctness checks from Section 3.2
    print("Confusion matrix total:", C.sum(), "(must be 21892)")
    print("Row sums == class counts:", bool((C.sum(axis=1) == counts).all()))

    print("\nConfusion matrix (rows = true, columns = predicted)")
    print("      " + "".join(f"{l:>8}" for l in LETTERS))
    for j in range(5):
        print(f"{LETTERS[j]:>4}  " + "".join(f"{C[j, k]:>8}" for k in range(5)))

    print("\nTable 4: test results")
    print(f"{'':<22}" + "".join(f"{l:>9}" for l in LETTERS))
    print(f"{'Per-class accuracy':<22}" + "".join(f"{v:>9.4f}" for v in pca))
    print(f"Test accuracy          : {acc:.4f}")
    print(f"Balanced accuracy      : {bal:.4f}")
    print(f"Always-'N' accuracy    : {always_n:.4f}")

    # Most common mistake for each class (useful for Task 6.2)
    print("\nMost frequent wrong prediction per true class")
    for j in range(5):
        off = C[j].copy()
        off[j] = 0
        k = int(off.argmax())
        print(f"  {LETTERS[j]} -> {LETTERS[k]}: {off[k]} of {C[j].sum()} rows")

    # Figure 4: confusion matrix heat-map, class names on both axes
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(NAMES, fontsize=8)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    ax.set_title("Test confusion matrix")
    thresh = C.max() / 2
    for j in range(5):
        for k in range(5):
            ax.text(k, j, str(C[j, k]), ha="center", va="center",
                    color="white" if C[j, k] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("\nsaved results/confusion_matrix.png")

    with open("results/test_results.json", "w") as f:
        json.dump({"test_accuracy": acc, "balanced_accuracy": bal,
                   "always_N_accuracy": float(always_n),
                   "per_class_accuracy": dict(zip(LETTERS, map(float, pca))),
                   "confusion_matrix": C.tolist()}, f, indent=2)


if __name__ == "__main__":
    main()

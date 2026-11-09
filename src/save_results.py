import json, os, math, shutil
import numpy as np
import torch

from load_data import load_raw
from prepare import prepare
from model import ECGNet, cross_entropy
from plot_beats import plot_one_per_class

LETTERS = ["N", "S", "V", "F", "Q"]
os.makedirs("results", exist_ok=True)

# ---------- Figure 1 (example beats) ----------
raw = load_raw("data")
plot_one_per_class(raw["train"], out_path="results/example_beats.png")

# ---------- Table 1: data ----------
data = prepare(verbose=False)
tr = raw["train"]; te = raw["test"]
sig = tr.iloc[:, :187].to_numpy()
ctr = np.bincount(tr.iloc[:, 187].to_numpy().astype(int), minlength=5)
cte = np.bincount(te.iloc[:, 187].to_numpy().astype(int), minlength=5)

# ---------- Table 2: parameters, initial loss, gradient check ----------
torch.manual_seed(0)
net = ECGNet()
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
with torch.no_grad():
    init_loss = cross_entropy(net(data["X_train"]), data["y_train"]).item()

X, y = data["X_train"][:64], data["y_train"][:64]
B = X.shape[0]
logits, acts = net(X, return_activations=True)
cross_entropy(logits, y).backward()
with torch.no_grad():
    Yh = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
    Y = torch.zeros(B, 5); Y[torch.arange(B), y] = 1.0
    dW3 = (Yh - Y).T @ acts[2] / B
    db3 = (Yh - Y).sum(0) / B
    diff_W = (dW3 - net.Ws[2].grad).abs().max().item()
    diff_b = (db3 - net.bs[2].grad).abs().max().item()

# ---------- Results from training and testing ----------
with open("results/train_summary.json") as f:
    s = json.load(f)
with open("results/test_results.json") as f:
    t = json.load(f)

a = str(s["chosen_alpha"]); h = s["runs"][a]["history"]; e = s["runs"][a]["best_epoch"]

md = []
md.append("# Results tables\n")
md.append("## Table 1: Data\n")
md.append("| Item | Value |\n|---|---|")
md.append(f"| Train / validation / test rows | {len(data['X_train'])} / {len(data['X_val'])} / {len(data['X_test'])} |")
md.append(f"| Min / max signal value (train file) | {sig.min()} / {sig.max()} |")
md.append("| Rows per class, train file (N, S, V, F, Q) | " + ", ".join(map(str, ctr)) + " |")
md.append("| Rows per class, test file (N, S, V, F, Q) | " + ", ".join(map(str, cte)) + " |\n")

md.append("## Table 2: Network and training\n")
md.append("| Item | Value |\n|---|---|")
md.append(f"| Trainable parameters | {n_params} |")
md.append(f"| Initial loss (log 5 = {math.log(5):.4f}) | {init_loss:.4f} |")
md.append(f"| Gradient check max difference W[3] / b[3] | {diff_W:.2e} / {diff_b:.2e} |")
md.append(f"| Chosen alpha | {a} |")
md.append(f"| Best epoch (lowest validation loss) | {e} |")
md.append(f"| Train / validation loss at best epoch | {h['train_loss'][e-1]:.4f} / {h['val_loss'][e-1]:.4f} |")
md.append(f"| Train / validation accuracy at best epoch | {h['train_acc'][e-1]:.4f} / {h['val_acc'][e-1]:.4f} |")
md.append(f"| Training time and hardware | {s['runs'][a]['time_sec']:.1f} s on {s['hardware']} |\n")

md.append("## Table 3: Learning-rate choice\n")
md.append("| alpha | Lowest validation loss | Validation accuracy at that epoch |\n|---|---|---|")
for k, r in s["runs"].items():
    ep = r["best_epoch"]
    md.append(f"| {k} | {r['best_val_loss']:.4f} | {r['history']['val_acc'][ep-1]:.4f} |")
md.append("")

md.append("## Table 4: Test results\n")
md.append("| | " + " | ".join(LETTERS) + " |\n|---|---|---|---|---|---|")
md.append("| Per-class accuracy | " + " | ".join(f"{t['per_class_accuracy'][l]:.4f}" for l in LETTERS) + " |")
md.append(f"\nTest accuracy: {t['test_accuracy']:.4f}  ")
md.append(f"Balanced accuracy: {t['balanced_accuracy']:.4f}  ")
md.append(f"Always-N accuracy: {t['always_N_accuracy']:.4f}\n")

md.append("### Test confusion matrix (rows = true class, columns = predicted)\n")
md.append("| true \\ pred | " + " | ".join(LETTERS) + " |\n|---|---|---|---|---|---|")
for j, row in enumerate(t["confusion_matrix"]):
    md.append(f"| {LETTERS[j]} | " + " | ".join(map(str, row)) + " |")

with open("results/tables.md", "w") as f:
    f.write("\n".join(md) + "\n")
print("saved results/tables.md")

# ---------- analysis.md: headings only, you write the answers ----------
if not os.path.exists("results/analysis.md"):
    with open("results/analysis.md", "w") as f:
        f.write("# Analysis\n\n"
                "## 6.1 Always-N model vs accuracy\n\n"
                "## 6.2 Lowest per-class accuracy and its confusion\n\n"
                "## 6.3 Loss curves: overfitting or not\n\n"
                "## 6.4 Zero initialisation and the He rule\n\n"
                "## 6.5 Momentum and weight decay\n\n"
                "## 6.6 Helping the rare class F\n")
    print("created results/analysis.md (write your answers in it)")

# ---------- Check that every required file exists ----------
required = {
    "Figure 1 example beats": "results/example_beats.png",
    "Figure 2 val loss, 3 alphas": "results/val_loss_alphas.png",
    "Figure 3a loss curves": "results/loss_curves.png",
    "Figure 3b accuracy curves": "results/acc_curves.png",
    "Figure 4 confusion matrix": "results/confusion_matrix.png",
    "Figure 5 front-end screenshot": "results/frontend_screenshot.png",
    "Saved model": "results/best_model.pt",
    "Tables": "results/tables.md",
    "Analysis": "results/analysis.md",
}
print("\nChecklist")
for name, path in required.items():
    print(f"  [{'OK' if os.path.exists(path) else 'MISSING'}] {name}: {path}")

# ---------- Zip for download ----------
folders = [d for d in ["src", "results", "app"] if os.path.isdir(d)]
for d in folders:
    pass
shutil.make_archive("project_backup", "zip", ".", None) if False else None
os.system("zip -rq project_backup.zip " + " ".join(folders))
print("\nsaved project_backup.zip containing: " + ", ".join(folders))

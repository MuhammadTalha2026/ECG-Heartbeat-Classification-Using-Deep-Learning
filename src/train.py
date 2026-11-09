import json
import time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prepare import prepare
from model import ECGNet, cross_entropy

# ---- Settings fixed by the assignment ------------------------------------
BATCH = 128
BETA = 0.9
LAMBDA = 1e-4
EPOCHS = 30
ALPHAS = [0.01, 0.1, 0.5]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(net, X, y, chunk=20000):
    """Mean cross-entropy and accuracy over a whole set (no gradients)."""
    total_loss, correct = 0.0, 0
    for i in range(0, len(X), chunk):
        xb, yb = X[i:i + chunk], y[i:i + chunk]
        logits = net(xb)
        total_loss += cross_entropy(logits, yb).item() * len(xb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def train(alpha, data, verbose=True):
    """4.1: train a fresh network (seed 0) for 30 epochs with momentum + weight decay."""
    X_train, y_train = data["X_train"].to(DEVICE), data["y_train"].to(DEVICE)
    X_val, y_val = data["X_val"].to(DEVICE), data["y_val"].to(DEVICE)

    torch.manual_seed(0)
    net = ECGNet().to(DEVICE)

    # Velocity for every parameter, starting at zero.
    # Weight decay is applied to the weights only, never to the biases.
    params = []
    for W in net.Ws:
        params.append((W, torch.zeros_like(W), True))
    for b in net.bs:
        params.append((b, torch.zeros_like(b), False))

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_epoch, best_state = float("inf"), None, None
    n = len(X_train)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(n)                       # shuffle every epoch
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH].to(DEVICE)
            loss = cross_entropy(net(X_train[idx]), y_train[idx])
            loss.backward()

            with torch.no_grad():
                for P, v, decay in params:
                    g = P.grad + LAMBDA * P if decay else P.grad
                    v.mul_(BETA).add_((1 - BETA) * g)  # v = beta*v + (1-beta)*g
                    P.sub_(alpha * v)                  # P = P - alpha*v
                    P.grad = None                      # clear gradient for next batch

        # Metrics after each epoch, on the full train and validation sets
        tr_loss, tr_acc = evaluate(net, X_train, y_train)
        va_loss, va_acc = evaluate(net, X_val, y_val)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)

        # Keep the parameters from the epoch with the lowest validation loss
        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if verbose:
            print(f"alpha={alpha:<5} epoch {epoch:2d}/{EPOCHS}  "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    return {"alpha": alpha, "hist": hist, "best_epoch": best_epoch,
            "best_val_loss": best_val, "best_state": best_state, "time_sec": elapsed}


def main():
    data = prepare(verbose=False)
    hw = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    print(f"Device: {hw}")

    runs = {}
    for a in ALPHAS:
        runs[a] = train(a, data)

    # ---- 4.2 Table 3 and the validation-loss figure ----------------------
    print("\nTable 3: learning-rate choice")
    print(f"{'alpha':<8}{'lowest val loss':>18}{'val acc at that epoch':>24}{'epoch':>8}")
    for a, r in runs.items():
        e = r["best_epoch"]
        print(f"{a:<8}{r['best_val_loss']:>18.4f}{r['hist']['val_acc'][e-1]:>24.4f}{e:>8}")

    epochs = list(range(1, EPOCHS + 1))
    plt.figure(figsize=(7, 4.5))
    for a, r in runs.items():
        plt.plot(epochs, r["hist"]["val_loss"], label=f"alpha = {a}")
    plt.xlabel("epoch"); plt.ylabel("validation loss")
    plt.title("Validation loss for the three learning rates")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/val_loss_alphas.png", dpi=150); plt.close()

    # ---- Choose alpha with the lowest validation loss --------------------
    best_a = min(runs, key=lambda a: runs[a]["best_val_loss"])
    r = runs[best_a]
    h, e = r["hist"], r["best_epoch"]
    print(f"\nChosen alpha = {best_a}")

    # ---- 4.3 Curves for the chosen alpha ---------------------------------
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, h["train_loss"], label="training loss")
    plt.plot(epochs, h["val_loss"], label="validation loss")
    plt.axvline(e, color="gray", linestyle="--", label=f"best epoch = {e}")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy loss")
    plt.title(f"Loss curves (alpha = {best_a})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/loss_curves.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, h["train_acc"], label="training accuracy")
    plt.plot(epochs, h["val_acc"], label="validation accuracy")
    plt.axvline(e, color="gray", linestyle="--", label=f"best epoch = {e}")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.title(f"Accuracy curves (alpha = {best_a})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/acc_curves.png", dpi=150); plt.close()

    # Save the parameters from the best epoch (lowest validation loss)
    torch.save(r["best_state"], "results/best_model.pt")
    print("saved results/best_model.pt")

    # ---- Table 2 values --------------------------------------------------
    print("\nTable 2 (training part)")
    print(f"Chosen alpha              : {best_a}")
    print(f"Best epoch                : {e}")
    print(f"Train / val loss at best  : {h['train_loss'][e-1]:.4f} / {h['val_loss'][e-1]:.4f}")
    print(f"Train / val acc at best   : {h['train_acc'][e-1]:.4f} / {h['val_acc'][e-1]:.4f}")
    print(f"Training time, hardware   : {r['time_sec']:.1f} s on {hw}")

    # Save everything needed for the report
    summary = {
        "chosen_alpha": best_a, "best_epoch": e, "hardware": hw,
        "runs": {str(a): {"best_epoch": x["best_epoch"], "best_val_loss": x["best_val_loss"],
                          "time_sec": x["time_sec"], "history": x["hist"]}
                 for a, x in runs.items()},
    }
    with open("results/train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

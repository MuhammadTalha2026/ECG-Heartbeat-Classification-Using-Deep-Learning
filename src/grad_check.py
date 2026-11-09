import math
import torch
from prepare import prepare
from model import ECGNet, cross_entropy


def main():
    data = prepare(verbose=False)
    X_train, y_train = data["X_train"], data["y_train"]

    torch.manual_seed(0)
    net = ECGNet()

    # ---- Initial loss on the whole training set (before any training) ----
    with torch.no_grad():
        init_loss = cross_entropy(net(X_train), y_train).item()
    print(f"Initial loss (full train set): {init_loss:.4f}   (log 5 = {math.log(5):.4f})")

    # ---- 3.2 Gradient check on the first 64 training rows ----
    X = X_train[:64]
    y = y_train[:64]
    B = X.shape[0]

    logits, acts = net(X, return_activations=True)
    A2 = acts[2]                                         # (B, 64), second hidden layer
    loss = cross_entropy(logits, y)
    loss.backward()
    print(f"Loss on first 64 rows: {loss.item():.4f}")

    # Hand-computed gradient (no autograd)
    with torch.no_grad():
        # softmax probabilities, written with logsumexp for numerical safety
        Y_hat = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))  # (B, 5)
        Y = torch.zeros(B, 5)
        Y[torch.arange(B), y] = 1.0                      # one-hot labels (B, 5)

        dW3 = (Y_hat - Y).T @ A2 / B                     # (5, 64)
        db3 = (Y_hat - Y).sum(dim=0) / B                 # (5,)

        diff_W = (dW3 - net.Ws[2].grad).abs().max().item()
        diff_b = (db3 - net.bs[2].grad).abs().max().item()

    print(f"Max |difference| for W[3]: {diff_W:.3e}")
    print(f"Max |difference| for b[3]: {diff_b:.3e}")
    print("PASS (< 1e-5)" if max(diff_W, diff_b) < 1e-5 else "FAIL: check your code")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn


class ECGNet(nn.Module):
    """187 -> 128 (ReLU) -> 64 (ReLU) -> 5 (logits), built from raw weight matrices."""

    def __init__(self, sizes=(187, 128, 64, 5)):
        super().__init__()
        self.sizes = sizes
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        for l in range(1, len(sizes)):
            n_in, n_out = sizes[l - 1], sizes[l]
            std = (2.0 / n_in) ** 0.5                    # He rule
            self.Ws.append(nn.Parameter(torch.randn(n_out, n_in) * std))
            self.bs.append(nn.Parameter(torch.zeros(n_out)))

    @staticmethod
    def relu(z):
        return torch.clamp(z, min=0)

    def forward(self, x, return_activations=False):
        A = x
        acts = [A]
        last = len(self.Ws) - 1
        for l, (W, b) in enumerate(zip(self.Ws, self.bs)):
            Z = A @ W.T + b
            if l < last:
                A = self.relu(Z)
                acts.append(A)
            else:
                A = Z                                    # logits
        if return_activations:
            return A, acts                               # acts = [x, A1, A2]
        return A


def cross_entropy(logits, y):
    """Mean softmax cross-entropy: L = -z_y + log(sum_j exp(z_j)).

    logits: (B, 5) float, y: (B,) long. Uses logsumexp, so no overflow.
    """
    B = logits.shape[0]
    z_true = logits[torch.arange(B), y]                  # z_y for each row
    L = -z_true + torch.logsumexp(logits, dim=1)         # per-example loss
    return L.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    net = ECGNet()
    total = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Trainable parameters:", total)
    for i, (W, b) in enumerate(zip(net.Ws, net.bs), start=1):
        print(f"Layer {i}: W{tuple(W.shape)} b{tuple(b.shape)}")

import math
import torch


def _init_weight(fan_in: int, fan_out: int, device: torch.device) -> torch.Tensor:
    """
    He-style std: sqrt(2 / fan_in) as required by the homework.
    """
    sigma = math.sqrt(2.0 / float(fan_in))
    W = torch.randn(fan_in, fan_out, dtype=torch.float32, device=device) * sigma
    W.requires_grad_(True)
    return W


def binary_classification(d: int, n: int, epochs: int = 10000, eta: float = 0.001):
    """
    Two-layer (actually 4 linear layers) binary classification with sigmoid activations.

    Inputs:
      d: number of features
      n: number of samples
      epochs: training epochs (default 10000)
      eta: learning rate (default 0.001)

    Output:
      W1, W2, W3, W4: trained weight tensors
      losses: torch.float32 tensor of length = epochs (cross-entropy loss history)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate X ~ N(0,1), shape (n, d)
    X = torch.randn(n, d, dtype=torch.float32, device=device)

    # Generate labels: Y = 1 if sum(features) > 2 else 0, shape (n, 1)
    Y = (X.sum(dim=1, keepdim=True) > 2.0).to(torch.float32)

    # Weight shapes: (d,48), (48,16), (16,32), (32,1)
    W1 = _init_weight(d, 48, device)
    W2 = _init_weight(48, 16, device)
    W3 = _init_weight(16, 32, device)
    W4 = _init_weight(32, 1, device)

    eps = 1e-12
    loss_hist = []

    for _ in range(int(epochs)):
        # Forward pass (sigmoid after each linear map)
        A1 = torch.sigmoid(X @ W1)      # (n, 48)
        A2 = torch.sigmoid(A1 @ W2)     # (n, 16)
        A3 = torch.sigmoid(A2 @ W3)     # (n, 32)
        Yhat = torch.sigmoid(A3 @ W4)   # (n, 1)

        # Binary cross-entropy (manual, so it's clear)
        loss = -(Y * torch.log(Yhat + eps) + (1.0 - Y) * torch.log(1.0 - Yhat + eps)).mean()
        loss_hist.append(loss.detach().item())

        # Backprop
        loss.backward()

        # Gradient descent update
        with torch.no_grad():
            W1 -= eta * W1.grad
            W2 -= eta * W2.grad
            W3 -= eta * W3.grad
            W4 -= eta * W4.grad

        # Zero gradients
        W1.grad.zero_()
        W2.grad.zero_()
        W3.grad.zero_()
        W4.grad.zero_()

    losses = torch.tensor(loss_hist, dtype=torch.float32)
    return W1, W2, W3, W4, losses

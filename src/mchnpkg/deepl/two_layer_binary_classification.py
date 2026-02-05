import torch


def binary_classification(d: int,
                          n: int,
                          epochs: int = 10000,
                          eta: float = 0.001,
                          seed: int = 0):
    """
    Binary classification with manual gradient descent using PyTorch autograd.

    Args:
        d: number of features
        n: number of samples
        epochs: number of training epochs (default 10000)
        eta: learning rate (default 0.001)
        seed: random seed (optional)

    Returns:
        W1, W2, W3, W4: trained weight matrices (torch.Tensor)
        losses: list of loss values per epoch (length = epochs)
    """

    # Device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # -----------------------------
    # Data: X and labels Y
    # -----------------------------
    X = torch.randn(n, d, dtype=torch.float32, device=device)

    # Label rule (edit this line if your homework specifies a different rule)
    # Example: label = 1 if sum of all features > 2 else 0
    Y = (X.sum(dim=1, keepdim=True) > 2).float()

    # -----------------------------
    # Weight initialization: N(0, 1/sqrt(n_in))
    # -----------------------------
    def init_W(n_in, n_out):
        std = 1.0 / (n_in ** 0.5)
        W = torch.randn(n_in, n_out, device=device, dtype=torch.float32) * std
        return torch.nn.Parameter(W)

    # Shapes from your earlier spec
    W1 = init_W(d, 48)
    W2 = init_W(48, 16)
    W3 = init_W(16, 32)
    W4 = init_W(32, 1)

    params = [W1, W2, W3, W4]

    # -----------------------------
    # Binary cross entropy loss
    # -----------------------------
    def bce(yhat, y, eps=1e-7):
        yhat = torch.clamp(yhat, eps, 1 - eps)
        return -(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat)).mean()

    # -----------------------------
    # Training loop
    # -----------------------------
    losses = []

    for _ in range(epochs):
        # Forward
        a1 = torch.sigmoid(X @ W1)
        a2 = torch.sigmoid(a1 @ W2)
        a3 = torch.sigmoid(a2 @ W3)
        yhat = torch.sigmoid(a3 @ W4)

        loss = bce(yhat, Y)
        losses.append(loss.item())

        # Backward
        loss.backward()

        # GD update
        with torch.no_grad():
            for p in params:
                p -= eta * p.grad
                p.grad.zero_()

    # Return trained weights as tensors + losses
    return W1.detach(), W2.detach(), W3.detach(), W4.detach(), losses

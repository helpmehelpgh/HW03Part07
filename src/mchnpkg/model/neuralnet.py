import torch
import torch.nn as nn

class TorchNet(nn.Module):
    """Two-hidden-layer network for binary classification."""
    def __init__(self, input_dim, output_dim=1,
                 hidden1=4, hidden2=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)   # logits
        return x

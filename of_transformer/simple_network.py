import torch.nn as nn


class DumbNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, mask=None):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

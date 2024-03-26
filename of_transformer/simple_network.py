import torch.nn as nn

class DumbNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            # nn.Linear(512, 512),
            # nn.GELU(),
            nn.Linear(512, 1),
        )

    def forward(self, x, mask=None):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Activation modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AGLU(nn.Module):
    """Unified activation function module from https://github.com/kostas1515/AGLU."""

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function."""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        lam = torch.clamp(self.lambd, min=0.0001)
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))

class GSigmoidV1(nn.Module):
    """
    output = 1 / [1 + exp(-alpha * (x - beta))]
    """
    def __init__(self, alpha_init=1.0, beta_init=0.0):
        super().__init__()
        self.alpha = alpha_init
        self.beta = beta_init

    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-self.alpha * (x - self.beta)))
    
    def __repr__(self):
        return f'GSigmoidV1(alpha={self.alpha}, beta={self.beta})'

class GeneralizedSigmoid(nn.Module):
    """
    output = 1 / [1 + exp(-alpha * (x - beta))]
    where alpha, beta are learnable parameters.
    """
    def __init__(self, alpha_init=1.0, beta_init=0.0):
        super().__init__()
        # Make alpha, beta trainable (learnable) parameters
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-self.alpha * (x - self.beta)))

class PELU(nn.Module):
    """Parametric Exponential Linear Unit (PELU) activation function."""
    def __init__(self, a=None, b=None):
        super().__init__()
        default_val = math.sqrt(0.1)
        a =  default_val if a is None else a 
        b = default_val  if b is None else b
        self.a = nn.Parameter(torch.tensor(a), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=True)

    def forward(self, x):
        a = torch.abs(self.a)
        b = torch.abs(self.b) 
        out = torch.where(x >= 0, a/b * x, a * (torch.exp(x / b) - 1))
        return out

# Hybrid Activation Unit (ReLU + SiLU)
class ReLUSiLU(nn.Module):
    def __init__(self, init_alpha=0.5):
        super(ReLUSiLU, self).__init__()
        # Alpha parameter balances the two activations
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x):
        relu_out = F.relu(x)
        silu_out = F.silu(x)  # Swish activation
        # Combine activations adaptively
        return self.alpha * relu_out + (1 - self.alpha) * silu_out

class ReLUELU(nn.Module):
    def __init__(self, alpha_init=1.0):
        super(ReLUELU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))  # Trainable alpha parameter

    def forward(self, x):
        return torch.where(x > 0, self.alpha * x, self.alpha * (torch.exp(x) - 1))

class Smish(nn.Module):
    """Applies the Smish activation function, a smooth approximation of ReLU."""
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    #@staticmethod
    def forward(self, x):
        return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))
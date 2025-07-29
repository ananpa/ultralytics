import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "LearnableFusion",
)

class LearnableFusion(nn.Module):
    """
    Fuse N feature maps with learnable non-negative weights that sum to 1
    (convex combination).  Two variants:
        mode = "scalar"   → one weight per input map
        mode = "spatial"  → 1x1 conv produces HxW weight maps
    """
    def __init__(self, n_inputs: int, mode: str = "scalar"):
        super().__init__()
        assert mode in ("scalar", "spatial")
        self.mode, self.n = mode, n_inputs
        if mode == "scalar": #  w_i  (N,)
            self.w = nn.Parameter(torch.zeros(n_inputs)) # will be softmaxed
        else: #  W_i  (N,1,1,1)
            self.w = nn.Parameter(torch.zeros(n_inputs, 1, 1, 1))

    def forward(self, xs): # xs = list[Tensor] length N
        assert len(xs) == self.n
        alpha = F.softmax(self.w, dim=0)     # convex weights
        return sum(a * x for a, x in zip(alpha, xs))

    def __repr__(self):
        return f'LearnableFusion(mode={self.mode}, w={self.w})'
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from torch.export import Dim

class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

# Define specific dimensions for dynamic shapes
inp1_dim0 = Dim("inp1_dim0")
_inp1_dim1 = Dim("_inp1_dim1", min=2, max=4611686018427387903) 
inp1_dim1 = 2 * _inp1_dim1

dynamic_shapes1 = {
    "x": {0: inp1_dim0, 1: inp1_dim1},
}

# Create a random input tensor
inp1 = torch.randn(4096, 6912)

# Export the model with dynamic shapes
exported_program = export(SiluAndMul(), (inp1,), dynamic_shapes=dynamic_shapes1)

# Save the exported program
torch.export.save(exported_program, 'silu_and_mul_exported_module.pt2')

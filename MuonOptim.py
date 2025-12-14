import torch
import torch.nn as nn
from Config import ModelConfig

class Muon(torch.optim.Optimizer):
  def __init__(self, params, lr = 0.02, momentum = 0.95, nesterov = True, ns_steps = 5):
    defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
    super().__init__(params, defaults)
  @torch.no_grad()
  def step(self):
    for group in self.param_groups:
      for p in group["params"]:
        if p.grad is None:
          continue

        g = p.grad
        state = self.state[p]

        if "momentum_buffer" not in state:
          state["momentum_buffer"] = torch.zeros_like(g)

        buf = state["momentum_buffer"]
        # Update momentum buffer: buf = momentum * buf + (1-momentum) * grad
        buf.lerp_(g, 1 - group["momentum"])
        # Apply Nesterov momentum if enabled, otherwise use standard momentum
        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
        # Apply zero-power normalization via Newton-Schulz iterations (make it close to orthonormal)
        g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
        # Update parameters with adaptive scaling based on parameter shape
        p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
        # Updates parameters with an adaptive learning rate that scales based on the parameter tensor's aspect ratio (height/width). 
        # For matrices where height > width, it increases the effective learning rate by √(height/width)

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
  assert G.ndim >= 2
  a, b, c = (3.4445, -4.7750, 2.0315)
  # [..., dim1, dim 2]
  X = G.bfloat16()
  # if dim1 > dim 2
  if G.size(-2) > G.size(-1):
    # [..., dim1, dim 2] -> [..., dim2, dim1]
    X = X.mT

  # [..., dim1, dim 2] assuming no transpose [..., dim2, dim 1] assuming transpose
  X = X/(X.norm(dim=(-2,-1), keepdim = True) + 1e-7)

  for _ in range(steps):
    # [..., dim1, dim 2] @ [..., dim2, dim 1] -> A: [..., dim1, dim 1] assuming no transpose  [..., dim2, dim 1] @ [..., dim1, dim 2] -> A: [..., dim2, dim 2] assuming transpose
    A = X@X.mT
    # [..., dim1, dim 1] + [..., dim1, dim 1] @ [..., dim1, dim 1] -> [..., dim1, dim 1] assuming no transpose [..., dim2, dim 2] + [..., dim2, dim 2] @ [..., dim2, dim 2] -> [..., dim2, dim 2] assuming transpose
    B = b * A + c * A @ A
    # [..., dim1, dim 2] + [..., dim1, dim 1] @ [..., dim1, dim 2] -> [..., dim1, dim 2] assuming no transpose [..., dim2, dim 1] + [..., dim2, dim 2] @ [..., dim2, dim 1] -> [..., dim2, dim 1] assuming transpose
    X = a * X + B @ X

 # This function is called again if the previous transpose was also called
  if G.size(-2) > G.size(-1):
    # with tran [..., dim2, dim 1] -> [..., dim1, dim 2]
    X = X.mT

  # [..., dim1, dim 2]
  return X



def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and
            'token_embedding' not in name and
            'norm' not in name and
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.muon_lr*0.1, weight_decay=config.weight_decay)

    return [muon_optimizer, adamw_optimizer]
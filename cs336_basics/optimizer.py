from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas:tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas" : betas, "eps" : eps, "weight_decay" : weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                t = state["step"]
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                p.data -= lr * weight_decay * p.data  # Apply weight decay
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # 一阶矩
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 二阶矩
                v.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2
                )
                # m_hat = m / (1 - beta1 ** t)
                # v_hat = v / (1 - beta2 ** t)
                
                p.data -= lr_t * m / (torch.sqrt(v) + eps)  # Apply weight decay

        return loss

def lr_cosine_schedule(
    t: int,
    a_max: float,
    a_min: float,
    tw: int,
    tc: int,
) -> float:
    if t < tw:
        return t * a_max / tw
    elif t > tc:
        return a_min
    else:
        return a_min + 0.5 * (1 + math.cos((t - tw) / (tc - tw) * math.pi)) * (a_max - a_min)



def grad_norm(parameters: Iterable[torch.nn.Parameter]):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm        


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float
):
    eps = 1e-6
    total_norm = grad_norm(parameters)
    factor = max_l2_norm / (total_norm + eps)
    if total_norm >= max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(factor)

    return 

def main():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e1)
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.


if __name__ == "__main__":
    main()





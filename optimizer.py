from typing import Iterable, Optional

import torch


def zeropower_via_newtonschulz5(gradient: torch.Tensor, steps: int) -> torch.Tensor:
    assert len(gradient.shape) == 2, "Muon expects 2D parameters"
    a, b, c = (3.4445, -4.7750, 2.0315)
    update = gradient.bfloat16()
    should_transpose = update.size(0) > update.size(1)
    if should_transpose:
        update = update.T

    update = update / (update.norm() + 1e-7)
    for _ in range(steps):
        gram = update @ update.T
        update = a * update + (b * gram + c * gram @ gram) @ update

    if should_transpose:
        update = update.T
    return update


if hasattr(torch, "compile"):
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for hidden 2D weight matrices."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        params = list(params)
        if not params:
            raise ValueError("Muon received no parameters")
        if not all(isinstance(param, torch.Tensor) for param in params):
            raise TypeError("Muon params must be tensors")
        if not all(param.ndim == 2 for param in params):
            raise ValueError("Muon should only receive 2D parameters")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buffer = state["momentum_buffer"]
                buffer.lerp_(grad, 1.0 - momentum)
                update = grad.lerp(buffer, momentum) if nesterov else buffer
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = max(1.0, param.size(0) / param.size(1)) ** 0.5
                param.add_(update, alpha=-lr * scale)
        return loss


def use_muon_for_parameter(name: str, parameter: torch.Tensor) -> bool:
    if parameter.ndim != 2:
        return False
    lowered_name = name.lower()
    if "embedding" in lowered_name or "embed" in lowered_name or "lm_head" in lowered_name:
        return False
    return True


def make_adam(
    params,
    lr: float,
    betas: tuple[float, float],
    fused: Optional[bool],
) -> torch.optim.Optimizer:
    all_params_cuda = True
    saw_param = False
    for group in params:
        group_params = group["params"] if isinstance(group, dict) else [group]
        for parameter in group_params:
            saw_param = True
            all_params_cuda = all_params_cuda and parameter.is_cuda
    use_fused = (
        torch.cuda.is_available() and saw_param and all_params_cuda
        if fused is None
        else fused
    )
    if use_fused:
        try:
            return torch.optim.Adam(params, lr=lr, betas=betas, fused=True)
        except TypeError:
            if fused:
                raise
            pass
    return torch.optim.Adam(params, lr=lr, betas=betas)


def build_adam_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    default_lr: float,
    lr_embed: Optional[float],
    lr_head: Optional[float],
    lr_scalar: Optional[float],
) -> list[dict]:
    groups = {
        "default": {"params": [], "lr": default_lr},
        "embed": {"params": [], "lr": default_lr if lr_embed is None else lr_embed},
        "head": {"params": [], "lr": default_lr if lr_head is None else lr_head},
        "scalar": {"params": [], "lr": default_lr if lr_scalar is None else lr_scalar},
    }
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        lowered_name = name.lower()
        if "embedding" in lowered_name or "embed" in lowered_name:
            groups["embed"]["params"].append(parameter)
        elif "lm_head" in lowered_name:
            groups["head"]["params"].append(parameter)
        elif parameter.ndim < 2:
            groups["scalar"]["params"].append(parameter)
        else:
            groups["default"]["params"].append(parameter)
    return [group for group in groups.values() if group["params"]]


def build_optimizers(
    model: torch.nn.Module,
    adam_lr: float,
    use_muon: bool = False,
    muon_lr: float = 1e-3,
    muon_momentum: float = 0.95,
    muon_ns_steps: int = 5,
    lr_embed: Optional[float] = None,
    lr_head: Optional[float] = None,
    lr_scalar: Optional[float] = None,
    fused_adam: Optional[bool] = None,
    adam_betas: tuple[float, float] = (0.9, 0.999),
) -> list[torch.optim.Optimizer]:
    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not use_muon:
        adam_groups = build_adam_param_groups(
            named_parameters,
            default_lr=adam_lr,
            lr_embed=lr_embed,
            lr_head=lr_head,
            lr_scalar=lr_scalar,
        )
        return [
            make_adam(
                adam_groups,
                lr=adam_lr,
                betas=adam_betas,
                fused=fused_adam,
            )
        ]

    muon_params = []
    adam_named_params = []
    for name, parameter in named_parameters:
        if use_muon_for_parameter(name, parameter):
            muon_params.append(parameter)
        else:
            adam_named_params.append((name, parameter))

    optimizers: list[torch.optim.Optimizer] = [
        Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            ns_steps=muon_ns_steps,
        )
    ]
    if adam_named_params:
        adam_groups = build_adam_param_groups(
            adam_named_params,
            default_lr=adam_lr,
            lr_embed=lr_embed,
            lr_head=lr_head,
            lr_scalar=lr_scalar,
        )
        optimizers.append(
            make_adam(
                adam_groups,
                lr=adam_lr,
                betas=adam_betas,
                fused=fused_adam,
            )
        )
    return optimizers

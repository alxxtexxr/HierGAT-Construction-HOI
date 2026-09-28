from typing import Optional

import torch
import torch.nn.functional as F


def binary_cross_entropy_loss(input, target, positive_class_weight=1, ignore_value=-1, reduction='mean'):
    """BCE loss for PyTorch tensors with optional value to be ignored.

    For the targets to be ignored (target == ignore_value), we multiply these targets and the corresponding predictions
    by zero so that we have zero loss on these terms.
    """
    mask = (target != ignore_value).float()
    num_nonmissing_elements = mask.sum().item()
    if num_nonmissing_elements == 0:
        return torch.tensor(0.0, dtype=input.dtype, device=input.device)
    if positive_class_weight > 1:
        input = torch.where(target == 1.0, input ** positive_class_weight, input)
    criterion = F.binary_cross_entropy(input * mask, target * mask, reduction=reduction)
    criterion *= input.numel() / num_nonmissing_elements
    return criterion


def budget_loss(input, target, ignore_value=-1, reduction='mean'):
    """Regularization loss to encourage a model to make the value in input zero.

    target has no effect on the loss computation and is only used to compute the valid entries in input. The keyword
    reduction also has no influence here and is kept only for signature compatibility with other PyTorch losses.
    """
    mask = (target != ignore_value).float()
    num_nonmissing_elements = mask.sum().item()
    if num_nonmissing_elements == 0:
        return torch.tensor(0.0, dtype=input.dtype, device=input.device)
    criterion = torch.mean(input * mask)
    criterion *= input.numel() / num_nonmissing_elements
    return criterion


def focal_loss(input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: Optional[float] = None,
               ignore_value: int = -1, reduction: str = 'mean') -> torch.Tensor:
    """Focal loss for multi-class classification as a drop-in replacement for F.nll_loss.

    Handles any combination of input/target shapes:
      - input (N, C), target (N,) or (N, 1) or (N, T)
      - input (N, C, T), target (N, T)
    """
    c = input.shape[1]

    if input.dim() > 2:
        # input: (N, C, T, ...) -> flatten to (N*T*..., C)
        input = input.permute(0, *range(2, input.dim()), 1).reshape(-1, c)
        # Match target to N*T*... samples
        if target.dim() > 1:
            target = target.reshape(-1)
        else:
            # target is (N,) but input is (N, C, T) — not expected, just flatten
            target = target.reshape(-1)
    else:
        # input is 2D (N, C). Ensure target is 1D (N,).
        if target.dim() > 1:
            # target is (N, T, ...) — reduce to (N,) via mode over non-batch dims
            target = target.squeeze(-1)  # (N, T, ...) -> at least (N, T)
            if target.dim() > 1:
                # mode over all non-batch dims
                for _ in range(target.dim() - 1):
                    target = torch.mode(target, dim=-1).values
    target = target.reshape(-1).long()

    # Create mask for valid targets
    valid_mask = (target != ignore_value)
    if not valid_mask.any():
        return torch.tensor(0.0, dtype=input.dtype, device=input.device)

    # Compute log probabilities on valid samples only
    valid_input = input[valid_mask]  # (num_valid, C)
    log_probs = F.log_softmax(valid_input, dim=1)
    probs = log_probs.exp()

    # Gather the probabilities for the target classes
    target_valid = target[valid_mask]  # (num_valid,)
    target_one_hot = F.one_hot(target_valid, num_classes=c).float()
    probs_target = (probs * target_one_hot).sum(dim=1)
    log_probs_target = (log_probs * target_one_hot).sum(dim=1)

    # Compute focal weight: (1 - p_t)^gamma
    focal_weight = (1 - probs_target) ** gamma

    # Apply alpha weighting if provided
    if alpha is not None:
        alpha_factor = torch.where(
            target_valid == 1,  # Assuming class 1 is the 'positive' class for binary
            alpha,
            1 - alpha
        )
        focal_weight = focal_weight * alpha_factor

    # Compute focal loss
    loss = -focal_weight * log_probs_target

    if reduction == 'none':
        # Reconstruct full-size tensor with zeros for ignored positions
        full_loss = torch.zeros(target.shape, dtype=input.dtype, device=input.device)
        full_loss[valid_mask] = loss
        return full_loss
    elif reduction == 'sum':
        return loss.sum()
    else:  # mean
        return loss.mean()


def focal_loss_for_nll(input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
                       ignore_value: int = -1, reduction: str = 'mean') -> torch.Tensor:
    """Focal loss wrapper with the same signature as F.nll_loss for use in multi_task_loss.

    This function has the same call signature as F.nll_loss (with ignore_index
    mapped to ignore_value) so it can be used as a drop-in replacement in
    the loss_functions tuple of multi_task_loss.

    Args:
        input: (N, C) tensor of log-probabilities.
        target: (N,) tensor of class indices.
        gamma: Focusing parameter. Default: 2.0.
        ignore_value: Class index to ignore (mapped from ignore_index in nll_loss).
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'.

    Returns:
        Scalar loss tensor.
    """
    return focal_loss(input, target, gamma=gamma, ignore_value=ignore_value, reduction=reduction)


def multi_task_loss(input: list, target: list, loss_functions: list, weight: Optional[list] = None,
                   ignore_value = -1, reduction: str = 'mean'):
    """Loss function for models with multiple losses."""
    if weight is None:
        weight = [1.0] * len(input)
    criteria = []
    for input_, target_, loss_function, w in zip(input, target, loss_functions, weight):
        if loss_function is F.nll_loss:
            target_ = torch.mode(target_.squeeze(-1), dim=1).values
            criterion = w * loss_function(input_, target_, ignore_index=ignore_value, reduction=reduction)
        elif loss_function is focal_loss_for_nll:
            target_ = torch.mode(target_.squeeze(-1), dim=1).values
            criterion = w * loss_function(input_, target_, ignore_value=ignore_value, reduction=reduction)
        else:
            criterion = w * loss_function(input_, target_, ignore_value=ignore_value, reduction=reduction)
        criteria.append(criterion)
    return criteria

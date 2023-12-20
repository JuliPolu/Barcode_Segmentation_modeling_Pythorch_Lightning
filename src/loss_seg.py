from itertools import filterfalse
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn
from torch.autograd import Variable

EPS = 1e-6


def dice_round(preds: torch.Tensor, trues: torch.Tensor) -> float:
    preds = preds.float()
    return soft_dice_loss(preds, trues)


def soft_dice_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    per_image: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    batch_size = outputs.size()[0] if per_image else 1
    flat_targets = targets.contiguous().view(batch_size, -1).float()
    flat_outputs = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(flat_outputs * flat_targets, dim=1)
    flat_outputs_sq = torch.sum(flat_outputs, dim=1)
    flat_targets_sq = torch.sum(flat_targets, dim=1)
    union = flat_outputs_sq + flat_targets_sq + eps
    return (1 - (2 * intersection + eps) / union).mean()


def jaccard(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    per_image: bool = False,
    non_empty: bool = False,
    min_pixels: int = 5,
    eps: float = 1e-3,
) -> torch.Tensor:

    batch_size = outputs.size()[0] if per_image else 1
    flat_targets = targets.contiguous().view(batch_size, -1).float()
    flat_outputs = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(flat_targets, dim=1)
    intersection = torch.sum(flat_outputs * flat_targets, dim=1)
    union = torch.sum(flat_outputs + flat_targets, dim=1) - intersection
    losses = 1 - (intersection + eps) / (union + eps)
    return _calculate_non_empty_loss(target_sum, losses, min_pixels) if non_empty else losses.mean()


def _calculate_non_empty_loss(target_sum: List[int], losses: List[float], min_pixels: int) -> float:
    non_empty_images, sum_loss = 0, 0
    for t_sum, loss in zip(target_sum, losses):
        if t_sum > min_pixels:
            sum_loss += loss
            non_empty_images += 1
    return sum_loss / non_empty_images if non_empty_images > 0 else 0


class DiceLoss(nn.Module):
    def __init__(self,         
        weight: Optional[torch.Tensor] = None,
        size_average: bool = True,
        per_image: bool = True,
    ) -> None:
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(
        self,
        input_x: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        
        probs = torch.sigmoid(input_x)
        return soft_dice_loss(probs, target, per_image=self.per_image)


class JaccardLoss(nn.Module):
    def __init__(  # noqa: WPS211
        self,
        weight: Optional[torch.Tensor] = None,
        size_average: bool = True,
        per_image: bool = False,
        non_empty: bool = False,
        apply_sigmoid: bool = False,
        min_pixels: int = 5,
    ) -> None:
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(
        self,
        input_x: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
    
        if self.apply_sigmoid:
            input_x = torch.sigmoid(input_x)
        return jaccard(
            input_x,
            target,
            per_image=self.per_image,
            non_empty=self.non_empty,
            min_pixels=self.min_pixels,
        )


class StableBCELoss(nn.Module):
    def forward(self, input_x, target):
        input_x = input_x.float().view(-1)
        target = target.float().view(-1)
        neg_abs = -input.abs()
        # todo check correctness
        loss = input_x.clamp(min=0) - input_x * target + (1 + neg_abs.exp()).log()  # noqa: WPS221
        return loss.mean()


class ComboLoss(nn.Module):  # noqa: WPS230
    def __init__(
        self,
        weights,
        per_image=False,
        channel_weights=None,
        channel_losses=None,
    ):
        super().__init__()
        self.weights = weights
        self.weight_values = {}
        self.channel_weights = channel_weights if channel_weights is not None else [1, 0.5, 0.5]
        self.channel_losses = channel_losses
        self._initialize_loss_functions(per_image)

    def forward(self, outputs, targets):
        loss = 0
        sigmoid_input = torch.sigmoid(outputs)
        for key, weight in self.weights.items():
            if weight:
                loss_contribution = self.calculate_loss_contribution(key, sigmoid_input, outputs, targets)
                self.weight_values[key] = loss_contribution
                loss += weight * loss_contribution
        return loss.clamp(min=1e-5)  # noqa: WPS432

    def calculate_loss_contribution(self, key, sigmoid_input, outputs, targets):
        if key in self.per_channel:
            return self.calculate_per_channel_loss(key, sigmoid_input, outputs, targets)

    def calculate_per_channel_loss(self, key, sigmoid_input, outputs, targets):
        return sum(
            self.evaluate_channel_loss(key, channel_index, sigmoid_input, outputs, targets)
            for channel_index in range(targets.size(1))
        )

    def evaluate_channel_loss(self, key, channel_index, sigmoid_input, outputs, targets):
        if self.channel_losses and key not in self.channel_losses[channel_index]:
            return 0
        input_tensor = (
            sigmoid_input[:, channel_index, ...]
            if key in self.expect_sigmoid
            else outputs[:, channel_index, ...]  # noqa: WPS221, E501
        )
        target_tensor = targets[:, channel_index, ...]
        return self.channel_weights[channel_index] * self.mapping[key](input_tensor, target_tensor)

    def calculate_general_loss(self, key, sigmoid_input, outputs, targets):
        input_tensor = sigmoid_input if key in self.expect_sigmoid else outputs
        return self.mapping[key](input_tensor, targets)

    def _initialize_loss_functions(self, per_image):
        self.bce = StableBCELoss()
        self.dice = DiceLoss(per_image=False)
        self.jaccard = JaccardLoss(per_image=False)
        self.lovasz = LovaszLoss(per_image=per_image)
        self.lovasz_sigmoid = LovaszLossSigmoid(per_image=per_image)
        self.focal = FocalLoss2d()
        self.mapping = {
            'bce': self.bce,
            'dice': self.dice,
            'focal': self.focal,
            'jaccard': self.jaccard,
            'lovasz': self.lovasz,
            'lovasz_sigmoid': self.lovasz_sigmoid,
        }
        self.expect_sigmoid = {'dice', 'focal', 'jaccard', 'lovasz_sigmoid'}
        self.per_channel = {'dice', 'jaccard', 'lovasz_sigmoid'}


def lovasz_grad(gt_sorted):
    """Lovasz gradient.

    Computes gradient of the Lovasz extension w.r.t sorted errors See Alg. 1 in paper.

    Returns:
        gradient of the Lovasz extension
    """
    pix = len(gt_sorted)
    gts = gt_sorted.sum()
    gts = gts.float()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard_loss = 1.0 - intersection / union
    if pix > 1:  # cover 1-pixel case
        jaccard_loss[1:pix] = jaccard_loss[1:pix] - jaccard_loss  # noqa: WPS221, WPS362
    return jaccard_loss


def lovasz_hinge(
    logits: torch.Tensor,
    labels: torch.Tensor,
    per_image: bool = True,
    ignore: Optional[int] = None,
) -> torch.Tensor:
    """Binary Lovasz hinge loss.

    Args:
        logits: [B, H, W] Variable, logits at each pixel (between -infty and +infty).
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1).
        per_image: compute the loss per image instead of per batch.
        ignore: void class id.

    Returns:
        The loss value.
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore),
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """Binary Lovasz hinge loss.

    Args:
        logits: [P] Variable, logits at each prediction (between -infty and +infty)
        labels: [P] Tensor, binary ground truth labels (0 or 1)

    Returns:
        The loss value
    """
    if len(labels) == 0:  # noqa: WPS507
        # only void pixels, the gradients should be 0
        return logits.sum() * 0  # noqa: WPS345
    signs = 2.0 * labels.float() - 1.0  # noqa: WPS432
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(nnf.relu(errors_sorted), Variable(grad))


def flatten_binary_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ignore: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flattens predictions in the batch (binary case). Removes labels equal to
    'ignore'.

    Args:
        scores: Predicted scores.
        labels: Ground truth labels.
        ignore: Label to ignore.

    Returns:
        Flattened valid scores and labels.
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_sigmoid(
    probas: torch.Tensor,
    labels: torch.Tensor,
    per_image: bool = False,
    ignore: Optional[int] = None
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss.

    probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
    labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
    only_present: average only on classes present in ground truth
    per_image: compute the loss per image instead of per batch
    ignore: void class labels

    Returns:
        torch.Tensor: The calculated Lovasz-Sigmoid loss.
    """
    if per_image:
        loss = mean(
            lovasz_sigmoid_flat(
                *flatten_binary_scores(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
            )
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = lovasz_sigmoid_flat(*flatten_binary_scores(probas, labels, ignore))
    return loss


def lovasz_sigmoid_flat(
    probas: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss.

    probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
    labels: [P] Tensor, ground truth labels (between 0 and C - 1)
    only_present: average only on classes present in ground truth

    Returns:
        torch.Tensor: The calculated Lovasz-Sigmoid loss.
    """
    fg = labels.float()
    errors = (Variable(fg) - probas).abs()
    errors_sorted, perm = torch.sort(errors, 0, descending=True)
    perm = perm.data
    fg_sorted = fg[perm]
    return torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted)))


def symmetric_lovasz(
    outputs: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """Compute symmetric Lovasz loss.

    Args:
        outputs: Predicted outputs.
        targets: Ground truth targets.

    Returns:
        torch.Tensor: The calculated symmetric Lovasz loss.
    """
    positive = lovasz_hinge(outputs, targets)
    negative = lovasz_hinge(-outputs, 1 - targets)
    return (positive + negative) / 2


def mean(
    lov: Iterable,
    ignore_nan: bool = False,
    empty_value: int = 0
) -> float:
    """Calculate the mean of the given values.

    Args:
        lov (iterable): Iterable containing the values.
        ignore_nan (bool): Whether to ignore NaN values.
        empty_value (int, str): Value to return if the input is empty.

    Returns:
        float: The mean of the given values.

    Raises:
        ValueError: If the input is empty and empty_value is 'raise'.
    """
    values_iter = iter(lov)
    if ignore_nan:
        values_iter = filterfalse(np.isnan, values_iter)

    total, count = 0, 0
    for value_lov in values_iter:
        total += value_lov
        count += 1

    if count == 0:
        if empty_value == 'raise':
            raise ValueError('Empty mean')
        return empty_value

    return total / count


class LovaszLoss(nn.Module):
    def __init__(self, ignore_index=255, per_image=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return symmetric_lovasz(outputs, targets)


class LovaszLossSigmoid(nn.Module):
    def __init__(self, ignore_index=255, per_image=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return lovasz_sigmoid(outputs, targets, per_image=self.per_image, ignore=self.ignore_index)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets, eps=1e-8):
        outputs = torch.sigmoid(logits)
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1.0 - eps)
        targets = torch.clamp(targets, eps, 1.0 - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-((1.0 - pt) ** self.gamma) * torch.log(pt)).mean()  # noqa: WPS221

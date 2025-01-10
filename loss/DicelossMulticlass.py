import torch
import torch.nn as nn


class DiceLossMulticlass(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        """
        Dice Loss for Multiclass Segmentation.
        Args:
            smooth (float): A small value to avoid division by zero.
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the loss.
        """
        super(DiceLossMulticlass, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Compute the Dice Loss for multiclass segmentation.
        Args:
            inputs (torch.Tensor): Predicted probabilities (logits) of shape [B, C, H, W].
            targets (torch.Tensor): Ground truth labels of shape [B, H, W], where each value is in [0, C-1].
        Returns:
            torch.Tensor: Dice loss value.
        """
        # Convert inputs to probabilities (softmax across classes)
        inputs = torch.softmax(inputs, dim=1)

        # One-hot encode the target labels to match the input shape
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)  # Shape: [B, C, H, W]

        # Exclude ignore_index (if specified) from the calculation
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            targets_one_hot = targets_one_hot * mask.unsqueeze(1)  # Exclude ignored regions

        # Compute intersection and union for each class
        intersection = torch.sum(inputs * targets_one_hot, dim=(2, 3))
        union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

        # Compute Dice score
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Take the average across all classes and batch
        dice_loss = 1 - dice_score.mean()

        return dice_loss
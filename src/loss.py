import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialCELoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=255, eps=1e-8):
        super(PartialCELoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, input, targets):
        labeled_mask = (targets != self.ignore_index).float()
        ce_loss = F.cross_entropy(input, targets, ignore_index=self.ignore_index, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        numerator = torch.sum(focal_loss * labeled_mask)
        denominator = torch.sum(labeled_mask) 

        return numerator / (denominator + self.eps)
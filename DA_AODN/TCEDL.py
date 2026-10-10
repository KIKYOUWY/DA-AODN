import torch.nn as nn
import torch

class Target_Contrast_Enhancement_Detection_Loss(nn.Module):
    def __init__(self, num_classes, feat_dim, device=None, alpha=1.0, beta=1.0):
        super(Target_Contrast_Enhancement_Detection_Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))
        self.alpha = alpha
        self.beta = beta

    def forward(self, features, labels):
        if features.ndim != 2 or features.shape[1] != self.feat_dim:
            raise ValueError("features must have shape (N, feat_dim)")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must have shape (N,) with integer class IDs")
        if labels.dtype != torch.long:
            raise ValueError("labels must have dtype torch.long")

        batch_size = features.shape[0]
        if batch_size == 0:
            return features.sum() * 0.0 + self.centers.sum() * 0.0
        if torch.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError("labels must be foreground class IDs in [0, num_classes)")

        distmat = (features[:, None, :] - self.centers[None, :, :]).square().sum(dim=-1)
        classes = torch.arange(self.num_classes, device=features.device)
        mask = labels[:, None].eq(classes[None, :])
        loss_close = distmat.masked_select(mask).sum() / batch_size
        loss_away = -torch.log(distmat.masked_select(~mask) + 1e-12).sum() / batch_size
        loss = self.alpha * loss_close + self.beta * loss_away
        return loss

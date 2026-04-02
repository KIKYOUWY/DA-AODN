import torch.nn as nn
import torch

class Target_Contrast_Enhancement_Detection_Loss(nn.Module):
    def __init__(self, num_classes, feat_dim, device, alpha=1.0, beta=1.0):
        super(Target_Contrast_Enhancement_Detection_Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        self.device = device
        self.alpha = alpha
        self.beta = beta

    def forward(self, features, labels):

        batch_size = features.size(0)
        distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(features, self.centers.t(), beta=1, alpha=-2)  # [batch_size, num_classes]

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.expand(batch_size, self.num_classes)  # [batch_size, num_classes]
        mask = labels.eq(classes.expand(batch_size, self.num_classes))  # [batch_size, num_classes]
        dist = distmat * mask.float()
        loss_close = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        inverse_mask = ~mask
        dist_away = distmat * inverse_mask.float()
        loss_away = -torch.log(dist_away + 1e-12)

        loss_away = loss_away.sum() / batch_size

        loss = self.alpha * loss_close + self.beta * loss_away
        return loss
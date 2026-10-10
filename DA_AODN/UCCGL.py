import torch.nn as nn
import torch
import torch.nn.functional as F
class AveragePooling(nn.Module):
    def __init__(self):
        super(AveragePooling, self).__init__()

    def forward(self, x):

        x = torch.mean(x, dim=(2))

        return x
class Unsupervised_Clear_Channel_Guided_Loss(nn.Module):
    def __init__(self):
        super(Unsupervised_Clear_Channel_Guided_Loss, self).__init__()
        self.GAP1 = AveragePooling()
        self.GAP2 = AveragePooling()
        self.KL = nn.KLDivLoss(reduction = 'batchmean')
    def forward(self, x, y):

        p = F.log_softmax(self.GAP1(x),1)

        q = F.softmax(self.GAP2(y),1)
        UCCGL = self.KL(p,q)

        return UCCGL


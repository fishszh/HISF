import torch
import torch.nn as nn


class ELLoss(nn.Module):
    def __init__(self, w=0.3, gamma=0.3):
        super().__init__()
        self.w = w
        self.gamma = gamma

    def forward(self, x, y):
        # print('x:', x.isnan().sum(), 'y:', y.isnan().sum())
        bcel = nn.BCEWithLogitsLoss()(x, y)
        x = x.sigmoid()
        dice = ((2 * x * y).sum((1,2,3,4)) + 1e-8) / ((x + y).sum((1,2,3,4)) + 1e-8)
        dicel = (- torch.log(dice)) ** self.gamma
        loss = self.w * bcel + dicel.mean()
        return loss
    
class MixLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        bcel = nn.BCEWithLogitsLoss()(x, y)
        x = x.sigmoid()
        dice = ((2 * x * y).sum((1,2,3,4)) + 1e-8) / ((x + y).sum((1,2,3,4)) + 1e-8)
        return 1 - dice.mean() + 0.5 * bcel



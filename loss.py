import torch
from torch import nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., from_logits=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, input, target):

        if self.from_logits:
            input = F.logsigmoid(input).exp()

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth))

#Taken from TORCHVISION.OPS.FOCAL_LOSS
class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            raise ValueError(f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")

        return loss
    
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
    
    def forward(self, input, target):
        #Assume the inputs are tensors of shape (batch, 1, L, W, H)
        # Calculate the change of value in the x, y, and z dimensions, for both positive and negative directions
        # To create a tensor of shape (batch, 6, L, W, H)
        # For each of input and target
        # Then calculate the mean squared error between the two

        dx_pos_input = input[:, :, :-1, :, :] - input[:, :, 1:, :, :]
        dx_neg_input = input[:, :, 1:, :, :] - input[:, :, :-1, :, :]
        dy_pos_input = input[:, :, :, :-1, :] - input[:, :, :, 1:, :]
        dy_neg_input = input[:, :, :, 1:, :] - input[:, :, :, :-1, :]
        dz_pos_input = input[:, :, :, :, :-1] - input[:, :, :, :, 1:]
        dz_neg_input = input[:, :, :, :, 1:] - input[:, :, :, :, :-1]
        grad_input = torch.cat([dx_pos_input, dx_neg_input, dy_pos_input, dy_neg_input, dz_pos_input, dz_neg_input], dim=1)

        dx_pos_target = target[:, :, :-1, :, :] - target[:, :, 1:, :, :]
        dx_neg_target= target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        dy_pos_target = target[:, :, :, :-1, :] - target[:, :, :, 1:, :]
        dy_neg_target = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        dz_pos_target = target[:, :, :, :, :-1] - target[:, :, :, :, 1:]
        dz_neg_target = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        grad_target = torch.cat([dx_pos_target, dx_neg_target, dy_pos_target, dy_neg_target, dz_pos_target, dz_neg_target], dim=1)

        # Then calculate the mean squared error between the two
        mse_loss = nn.MSELoss()
        loss = mse_loss(grad_input, grad_target)

        return loss

        
class IsodoseLoss(nn.Module):
    def __init__(self):
        super(IsodoseLoss, self).__init__()
        self.thresholds = [(0, 20), (20, 40), (40, 55), (55, 65), (65, 75), (75, torch.inf)]

    def forward(self, input, target):
        # Input and target are tensors of shape (batch, 1, L, W, H)

        input_categorical = torch.stack([(input > t[0]) & (input <= t[1]) for t in self.thresholds], dim=1).float()
        target_categorical = torch.stack([(target > t[0]) & (target <= t[1]) for t in self.thresholds], dim=1).float()

        # Convert target to class indices
        target_indices = torch.argmax(target_categorical, dim=1)

        # Calculate cross entropy loss"
        cce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 100, 100, 100, 100, 100]).type(torch.FloatTensor).to(input.device))
        loss = cce(input_categorical, target_indices)

        return loss

    
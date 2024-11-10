import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional


class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class NEWLoss(nn.Module):
    def __init__(self, reduction = None, base = "MSE", distance = "KL", Lambda = 0.01, temperature = 1, temp_to = "both"):
        super().__init__()
        self.base = base
        self.distance = distance
        self.Lambda = Lambda
        self.reduction = reduction
        self.temperature = temperature
        self.temp_to = temp_to
    def forward(self, predicted, actual, epsilon=1e-8): 
        # Add epsilon to avoid division by zero
        if self.base == "MSE":
            base = nn.MSELoss()
        elif self.base == "MAE":
            base = nn.L1Loss()
        if self.temp_to == "both":
            p = torch.softmax(predicted/self.temperature, -1) + epsilon
            q = torch.softmax(actual/self.temperature, -1) + epsilon
        elif self.temp_to == "true":
            p = torch.softmax(predicted, -1) + epsilon
            q = torch.softmax(actual/self.temperature, -1) + epsilon
        else:
            p = torch.softmax(predicted/self.temperature, -1) + epsilon
            q = torch.softmax(actual, -1) + epsilon
        if self.distance == "KL":
            distance = torch.sum(p * torch.log(p / q))
            # print("base(predicted, actual), distance :" , base(predicted, actual), distance)
        elif self.distance == "EM":
            cdf_p = torch.cumsum(p, dim=-1)
            cdf_q = torch.cumsum(q, dim=-1)
            distance = torch.sum(torch.abs(cdf_p - cdf_q))
            
        return  base(predicted, actual) + self.Lambda*distance
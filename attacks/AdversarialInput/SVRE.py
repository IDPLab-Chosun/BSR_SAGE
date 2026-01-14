import torch
from .AdversarialInputBase import AdversarialInputAttacker
from typing import Callable, List, Iterable
from attacks.utils import *
from .utils import cosine_similarity
from torch import nn
import random
from torchvision import transforms
import numpy as np
from scipy import stats as st
from torch import Tensor
def custom_clamp(x, min_tensor, max_tensor):
    return torch.max(torch.min(x, max_tensor), min_tensor)

class MI_SVRE(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 1.6 / 255,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1,
                 *args, **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_SVRE, self).__init__(model, *args, **kwargs)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            self.begin_attack(x.clone().detach())
            # first calculate the ensemble gradient
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device))
            loss = self.criterion(logit, y.to(model.device)).to(self.device)
            loss.backward()
            ensemble_gradient: Tensor = x.grad.clone()
            x.grad = None
            x.requires_grad = False
            for model in self.models:
                # with original grad
                self.original.requires_grad = True
                loss = self.criterion(model(self.original.to(model.device)), y.to(model.device)).to(self.device)
                loss.backward()
                original_grad = self.original.grad.clone()
                self.original.requires_grad = False
                self.original.grad = None
                # normal
                x.requires_grad = True
                loss = self.criterion(model(x.to(model.device)), y.to(model.device)).to(self.device)
                loss.backward()
                grad: Tensor = x.grad
                x.requires_grad = False
                grad = grad - (original_grad - ensemble_gradient)
                # update
                if self.targerted_attack:
                    momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                    x += self.step_size * momentum.sign()
                else:
                    momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                    x += self.step_size * momentum.sign()
                #x = clamp(x)
                x = custom_clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(final_momentum=momentum)
            x = custom_clamp(x, original_x - self.epsilon, original_x + self.epsilon)
        return x

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin

    @torch.no_grad()
    def end_attack(self, final_momentum: torch.tensor):
        '''
        '''
        self.outer_momentum = self.mu * self.outer_momentum + final_momentum
        x = self.original + self.step_size * self.outer_momentum.sign()
        del self.original
        return x

import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker
import torch.nn.functional as F
from timm.models import create_model

import sys
import os
import methods

def custom_clamp(x, min_tensor, max_tensor):
    return torch.max(torch.min(x, max_tensor), min_tensor)
    

class MI_SAM_1(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 reverse_step_size: float = 16 / 255 / 10,
                 *args, **kwargs
                 ):
        self.models = model
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_SAM_1, self).__init__(model, *args, **kwargs)
        self.reverse_step_size = reverse_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            ori_x = x.clone()
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            print(f"loss:{loss}")
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x += self.reverse_step_size * grad.sign()
            # x.grad = None
            # --------------------------------------------------------------------------------#
            # second step
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            x.mul_(0).add_(ori_x)
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            x = clamp(x)
            x = custom_clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x
class MI_SAM_ori(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 reverse_step_size: float = 16 / 255 / 10,
                 *args, **kwargs
                 ):
        self.models = model
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_SAM_ori, self).__init__(model, *args, **kwargs)
        self.reverse_step_size = reverse_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            ori_x = x.clone()
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.grad=None

            grad_store=[]
            for model in self.models:
                loss = self.criterion(model(x.to(model.device)), y.to(model.device))
                loss.backward()
                
                grad_store.append(x.grad)
                
            avg_grad = sum(grad_store)/len(grad_store)
            grad = -avg_grad-grad
            

            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * grad.sign()
            # x.grad = None
            # --------------------------------------------------------------------------------#
            # second step
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            x.mul_(0).add_(ori_x)
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            #x = clamp(x)
            x = custom_clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x




class MI_SAM(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 reverse_step_size: float = 16 / 255 / 10,
                 *args, **kwargs
                 ):
        self.models = model
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_SAM, self).__init__(model, *args, **kwargs)
        self.reverse_step_size = reverse_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        attack_method1 = getattr(methods, "TGR")('vit_base_patch16_224')
        attack_method2 = getattr(methods, "TGR")("visformer_small")

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            ori_x = x.clone()
            x.requires_grad = True
            logit = 0
            for model in self.models:
                logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * grad.sign()
            # x.grad = None
            # --------------------------------------------------------------------------------#
            # second step
            x.requires_grad = True
            logit = 0
            for model in self.models:
                if type(model).__name__ =="VisionTransformer":
                    #attack_method = getattr(methods, "TGR")("vit_base_patch16_224")
                    grad = attack_method1((aug_x.to(model.device).detach()), y.to(model.device).detach())
####
                elif type(model).__name__ =="Visformer":
                #    #attack_method = getattr(methods, "TGR")("visformer_small")
                    grad = attack_method2((aug_x.to(model.device).detach()), y.to(model.device).detach())
####
                #logit += model(x.to(model.device)).to(x.device)
            loss = self.criterion(logit, y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            x.mul_(0).add_(ori_x)
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + grad / torch.norm(grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            #x = clamp(x)
            x = custom_clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x





#논문에서 쓴거
class MI_SAM_avg(AdversarialInputAttacker):
   def __init__(self, model: List[nn.Module],
                total_step: int = 10, random_start: bool = False,
                step_size: float = 16 / 255 / 5,
                criterion: Callable = nn.CrossEntropyLoss(),
                targeted_attack=False,
                mu: float = 1,
                reverse_step_size: float = 16 / 255 / 10,
                *args, **kwargs
                ):
       self.models = model
       self.random_start = random_start
       self.total_step = total_step
       self.step_size = step_size
       self.criterion = criterion
       self.targerted_attack = targeted_attack
       self.mu = mu
       super(MI_SAM_avg, self).__init__(model, *args, **kwargs)
       self.reverse_step_size = reverse_step_size


   def perturb(self, x):
       x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
       x = clamp(x)
       return x


   def attack(self, x, y, ):
       N = x.shape[0]
       original_x = x.clone()
       momentum = torch.zeros_like(x)
       grad_store = []
       grad_store2=[]
       if self.random_start:
           x = self.perturb(x)


       attack_method1 = getattr(methods, "TGR")('vit_base_patch16_224')
       attack_method2 = getattr(methods, "TGR")("visformer_small")


       for _ in range(self.total_step):
           # --------------------------------------------------------------------------------#
           # first step 각 모델의 평균 그레디언트 찾기
           ori_x = x.clone()
           for model in self.models:
               x.requires_grad = True
               loss = self.criterion(model(x.to(model.device)), y.to(model.device))
               loss.backward()
              
               grad_store.append(x.grad)
           avg_grad = sum(grad_store)/len(grad_store)


          
           x.requires_grad = False
           if self.targerted_attack:
               pass
           else:
               x -= self.reverse_step_size * avg_grad.sign()
           # x.grad = None
           # --------------------------------------------------------------------------------#
           # second step
           x.requires_grad = True
           logit = 0
           for model in self.models:
               x.requires_grad = True
               loss = self.criterion(model(x.to(model.device)), y.to(model.device))
               loss.backward()
               if type(model).__name__ =="VisionTransformer":
                   #attack_method = getattr(methods, "TGR")("vit_base_patch16_224")
                   grad = attack_method1((x.to(model.device).detach()), y.to(model.device).detach())
####
               elif type(model).__name__ =="Visformer":
               #    #attack_method = getattr(methods, "TGR")("visformer_small")
                   grad = attack_method2((x.to(model.device).detach()), y.to(model.device).detach())
               grad_store2.append(grad)
           avg_grad = sum(grad_store2)/len(grad_store)


           x.requires_grad = False
           x.mul_(0).add_(ori_x)
           # update
           if self.targerted_attack:
               momentum = self.mu * momentum - avg_grad / torch.norm(avg_grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
               x += self.step_size * momentum.sign()
           else:
               momentum = self.mu * momentum + avg_grad / torch.norm(avg_grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
               x += self.step_size * momentum.sign()
           #x = clamp(x)
           x =custom_clamp(x, original_x - self.epsilon, original_x + self.epsilon)


       return x


class MI_SAM_avg보다_성능좋지만_제안하는방법과좀어긋나는부분이_있음(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 reverse_step_size: float = 16 / 255 / 10,
                 *args, **kwargs
                 ):
        self.models = model
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_SAM_avg, self).__init__(model, *args, **kwargs)
        self.reverse_step_size = reverse_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        
        
        if self.random_start:
            x = self.perturb(x)

        attack_method1 = getattr(methods, "TGR")('vit_base_patch16_224')
        attack_method2 = getattr(methods, "TGR")("visformer_small")

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step 각 모델의 평균 그레디언트 찾기
            ori_x = x.clone()
            grad_store = []
            for model in self.models:
                
                x.requires_grad = True
                loss = self.criterion(model(x.to(model.device)), y.to(model.device))
                loss.backward()
                
                grad_store.append(x.grad)
            avg_grad = sum(grad_store)/len(grad_store)

            
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * avg_grad.sign()
            # x.grad = None
            # --------------------------------------------------------------------------------#
            # second step
            x.requires_grad = True
            logit = 0
            grad_store2=[]
            for model in self.models:
                x.requires_grad = True
                loss = self.criterion(model(x.to(model.device)), y.to(model.device))
                loss.backward()
                if type(model).__name__ =="VisionTransformer":
                    #attack_method = getattr(methods, "TGR")("vit_base_patch16_224")
                    grad = attack_method1((x.to(model.device).detach()), y.to(model.device).detach())
####
                elif type(model).__name__ =="Visformer":
                #    #attack_method = getattr(methods, "TGR")("visformer_small")
                    grad = attack_method2((x.to(model.device).detach()), y.to(model.device).detach())
                grad_store2.append(grad)
            avg_grad = sum(grad_store2)/len(grad_store2)

            x.requires_grad = False
            x.mul_(0).add_(ori_x)
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - avg_grad / torch.norm(avg_grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + avg_grad / torch.norm(avg_grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            #x = clamp(x)
            x =custom_clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x



class MI_SAM_avg_egsam(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 reverse_step_size: float = 16 / 255 / 10,
                 *args, **kwargs
                 ):
        self.models = model
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_SAM_avg_egsam, self).__init__(model, *args, **kwargs)
        self.reverse_step_size = reverse_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        grad_store = []
        grad_store2=[]
        if self.random_start:
            x = self.perturb(x)

        

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step 각 모델의 평균 그레디언트 찾기
            ori_x = x.clone()
            for model in self.models:
                x.requires_grad = True
                loss = self.criterion(model(x.to(model.device)), y.to(model.device))
                loss.backward()
                
                grad_store.append(x.grad)
            avg_grad = sum(grad_store)/len(grad_store)

            
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * avg_grad.sign()
            # x.grad = None
            # --------------------------------------------------------------------------------#
            # second step
            x.requires_grad = True
            logit = 0
            for model in self.models:
                x.requires_grad = True
                loss = self.criterion(model(x.to(model.device)), y.to(model.device))
                loss.backward()
                
                grad_store2.append(x.grad)
            avg_grad = sum(grad_store2)/len(grad_store)

            x.requires_grad = False
            x.mul_(0).add_(ori_x)
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - avg_grad / torch.norm(avg_grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + avg_grad / torch.norm(avg_grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            #x = clamp(x)
            x =custom_clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

class MI_SAM_avg_(AdversarialInputAttacker):
    def __init__(self, model: List[nn.Module],
                 total_step: int = 10, random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu: float = 1,
                 reverse_step_size: float = 16 / 255 / 10,
                 *args, **kwargs
                 ):
        self.models = model
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        super(MI_SAM_avg_, self).__init__(model, *args, **kwargs)
        self.reverse_step_size = reverse_step_size

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        momentum = torch.zeros_like(x)
        grad_store = []
        grad_store2=[]
        if self.random_start:
            x = self.perturb(x)

        

        for _ in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step 각 모델의 평균 그레디언트 찾기
            ori_x = x.clone()
            for model in self.models:
                x.requires_grad = True
                loss = self.criterion(model(x.to(model.device)), y.to(model.device))
                loss.backward()
                
                grad_store.append(x.grad)
            avg_grad = sum(grad_store)/len(grad_store)

            
            x.requires_grad = False
            if self.targerted_attack:
                pass
            else:
                x -= self.reverse_step_size * avg_grad.sign()
            # x.grad = None
            # --------------------------------------------------------------------------------#
            # second step
            x.requires_grad = True
            logit = 0
            for model in self.models:
                x.requires_grad = True
                loss = self.criterion(model(x.to(model.device)), y.to(model.device))
                loss.backward()
                
                grad_store2.append(x.grad)
            avg_grad = sum(grad_store2)/len(grad_store)

            x.requires_grad = False
            x.mul_(0).add_(ori_x)
            # update
            if self.targerted_attack:
                momentum = self.mu * momentum - avg_grad / torch.norm(avg_grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            else:
                momentum = self.mu * momentum + avg_grad / torch.norm(avg_grad.reshape(N, -1), p=1, dim=1).view(N, 1, 1, 1)
                x += self.step_size * momentum.sign()
            #x = clamp(x)
            x =custom_clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x



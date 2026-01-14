import torch
from attacks.utils import *
from torch import nn
from typing import Callable, List
from .AdversarialInputBase import AdversarialInputAttacker
import numpy as np
from scipy import stats as st


class MI_TI_FGSM(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 epsilon: float = 16 / 255,
                 random_start: bool = False,
                 step_size: float = None,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack: bool = False,
                 mu: float = 1.0,   # 인터페이스 호환용(미사용)
                 *args, **kwargs):
        """
        TI-FGSM attacker (Translation-Invariant, Momentum 제거).

        Args
        ----
        model : List[nn.Module]s
            Surrogate models.
        total_step : int
            # of iterative steps (T).
        epsilon : float
            L_inf perturbation budget.
        random_start : bool
            If True, start from a random point within epsilon-ball.
        step_size : float
            Per-step update size.  If None, defaults to epsilon / total_step.
        criterion : Callable
            Loss function (default: CrossEntropy).
        targeted_attack : bool
            True → targeted attack, False → untargeted.
        mu : float
            (미사용) 기존 인터페이스 유지용.
        *args, **kwargs :
            Passed straight to AdversarialInputAttacker.
        """
        super().__init__(model, epsilon=epsilon, *args, **kwargs)

        self.total_step = total_step
        self.epsilon = epsilon
        self.random_start = random_start
        self.step_size = step_size if step_size is not None else epsilon / total_step
        self.criterion = criterion
        self.targeted_attack = targeted_attack
        self.mu = mu  # 미사용

        # Pre-built Gaussian kernel conv layer (no grad)
        self.conv = self._gkern_conv().to(self.device)
        self.conv.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Attack pipeline
    # ------------------------------------------------------------------ #
    def _random_perturb(self, x: torch.Tensor) -> torch.Tensor:
        """Random start within L_inf-epsilon box."""
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        return clamp(x)

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        TI-FGSM:
          g = ∇_x L(f(x), y)
          g_ti = GaussianDepthwiseConv(g)   # translation-invariant smoothing
          x <- x ± step_size * sign(g_ti)
          x <- Π_{B_inf(x0, ε)}(x)
        """
        x = x.clone().to(self.device)
        y = y.to(self.device)
        orig = x.clone()

        if self.random_start:
            x = self._random_perturb(x)
            x = clamp(x, orig - self.epsilon, orig + self.epsilon)

        for _ in range(self.total_step):
            x.requires_grad_(True)
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()

            # 여러 surrogate 모델의 로짓 합산
            logit_sum = 0
            for m in self.models:
                logit_sum += m(x.to(m.device)).to(x.device)

            loss = self.criterion(logit_sum, y)
            loss.backward()
            grad = x.grad.detach()
            x.requires_grad_(False)

            # Translation-invariant smoothing (가우시안 depthwise conv)
            grad_ti = self.conv(grad)

            # FGSM-style 업데이트 (모멘텀 제거)
            direction = -1 if self.targeted_attack else 1
            x = x + direction * self.step_size * grad_ti.sign()

            # 픽셀 범위 & L_inf 프로젝션
            x = clamp(x)                                   # [0,1] box
            x = clamp(x, orig - self.epsilon, orig + self.epsilon)  # L_inf ball
            x = x.detach()

        return x

    # ------------------------------------------------------------------ #
    # Helper: depth-wise Gaussian blur conv
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gkern_conv(klen: int = 15, sigma: float = 3) -> nn.Conv2d:
        """Depth-wise conv layer with frozen 2-D Gaussian kernel."""
        ax = np.linspace(-sigma, sigma, klen)
        gauss1d = st.norm.pdf(ax)
        kernel_raw = np.outer(gauss1d, gauss1d)
        kernel = (kernel_raw / kernel_raw.sum()).astype(np.float32)

        tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)      # (1,1,k,k)
        tensor = tensor.repeat(3, 1, 1, 1)                               # (3,1,k,k)

        conv = nn.Conv2d(3, 3, kernel_size=klen, padding=klen // 2,
                         groups=3, bias=False)
        conv.weight.data = tensor
        conv.weight.requires_grad = False
        return conv


# Quick test
if __name__ == '__main__':
    # Dummy models / data for smoke-test only
    dummy_net = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(16 * 224 * 224, 10)
    )
    attacker = MI_TI_FGSM([dummy_net], total_step=1)  # minimal test
    x_dummy = torch.rand(2, 3, 224, 224)
    y_dummy = torch.tensor([0, 1])
    adv = attacker.attack(x_dummy, y_dummy)
    print('Δ L_inf:', (adv - x_dummy).abs().max().item())

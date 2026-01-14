import torch
import torch.nn.functional as F
from torch import nn
from typing import Callable, List

from attacks.utils import *  # clamp, etc.
from .AdversarialInputBase import AdversarialInputAttacker


class DI_MI_FGSM(AdversarialInputAttacker):
    """
    DI^2-FGSM 구현 (클래스명은 호환성을 위해 DI_MI_FGSM 유지)
    - 모멘텀(MI) 제거
    - 매 스텝: 입력 다양화(무작위 리사이즈+패딩)를 확률 p로 적용 후 I-FGSM 업데이트
    - 다중 surrogate 모델의 로짓 합산 지원
    """

    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 16 / 255 / 10,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack: bool = False,
                 mu: float = 1.0,  # 인터페이스 유지용 (사용하지 않음)
                 diversity_prob: float = 0.5,
                 resize_rate: float = 0.9,
                 *args, **kwargs):
        """
        Args:
            model: surrogate 모델 리스트
            total_step: 반복 횟수 (T)
            random_start: 랜덤 스타트 사용 여부
            step_size: 스텝 크기 (alpha)
            criterion: 손실 함수
            targeted_attack: 타깃 공격 여부 (True면 반대 방향 업데이트)
            mu: (미사용) 기존 MI 인터페이스 호환을 위해 유지
            diversity_prob: 입력 다양화 적용 확률 p
            resize_rate: 리사이즈 하한 비율 r (예: 0.9이면 [0.9H, H] 범위에서 랜덤)
        """
        super(DI_MI_FGSM, self).__init__(model, *args, **kwargs)
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targeted_attack = targeted_attack
        self.targerted_attack = targeted_attack  # 오탈자 호환
        self.mu = mu

        # DI^2-FGSM 하이퍼파라미터
        self.diversity_prob = diversity_prob
        self.resize_rate = resize_rate

        self.init()

    @staticmethod
    def _model_device(m: nn.Module) -> torch.device:
        # model.device 속성이 없을 수 있으므로 안전하게 추출
        return getattr(m, "device", next(m.parameters()).device)

    def input_diversity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Diverse Inputs 변환:
        - 확률 p로 무작위 리사이즈(>= resize_rate * size) 후 제로 패딩으로 원래 크기에 맞춤
        - 확률 (1-p)로는 항등 변환
        """
        if self.diversity_prob <= 0.0:
            return x
        if torch.rand(1, device=x.device).item() > self.diversity_prob:
            return x

        B, C, H, W = x.shape
        # 리사이즈 크기 샘플 (H,W 각각 독립 샘플; 필요시 동일 크기 사용 가능)
        new_h_low = max(1, int(self.resize_rate * H))
        new_w_low = max(1, int(self.resize_rate * W))
        new_h = int(torch.randint(low=new_h_low, high=H + 1, size=(1,)).item())
        new_w = int(torch.randint(low=new_w_low, high=W + 1, size=(1,)).item())

        # 연속적 보간(미분가능)
        x_resized = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # 패딩 위치 무작위
        pad_top = int(torch.randint(low=0, high=H - new_h + 1, size=(1,)).item())
        pad_left = int(torch.randint(low=0, high=W - new_w + 1, size=(1,)).item())
        pad_bottom = H - new_h - pad_top
        pad_right = W - new_w - pad_left

        # F.pad의 pad 순서는 (left, right, top, bottom)
        x_padded = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
        return x_padded

    def perturb(self, x: torch.Tensor) -> torch.Tensor:
        """
        랜덤 스타트: [-epsilon, epsilon] 균등 노이즈 추가 후 픽셀 클램프
        """
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)  # [0,1] 등 픽셀 범위로 클램프 (유틸 함수 가정)
        return x

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        DI^2-FGSM:
            x_{t+1} = Π_{B(x0, ε)}( x_t ± α * sign(∇_x L(f(T(x_t)), y)) )
        - T: input_diversity (확률적 리사이즈+패딩)
        - ±: untargeted는 +, targeted는 -
        """
        x = x.detach()
        y = y.to(x.device)
        original_x = x.clone().detach()

        if self.random_start:
            x = self.perturb(x)
            # 필요시 L_inf 볼 내부 보장 (랜덤 스타트 자체가 보장하지만 중복 방어)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        for _ in range(self.total_step):
            x.requires_grad_(True)

            # 입력 다양화
            aug_x = self.input_diversity(x)

            # 여러 모델 로짓 합산
            logit = None
            for m in self.models:
                m_dev = self._model_device(m)
                out = m(aug_x.to(m_dev))
                out = out.to(x.device)
                logit = out if logit is None else (logit + out)

            loss = self.criterion(logit, y)

            # 기존 grad 정리 후 역전파
            if x.grad is not None:
                x.grad.detach_()
                x.grad.zero_()
            loss.backward()
            grad = x.grad.detach()

            # FGSM 업데이트
            if self.targeted_attack:
                x = x - self.step_size * grad.sign()
            else:
                x = x + self.step_size * grad.sign()

            # 픽셀 범위 & L_inf 프로젝션
            x = clamp(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

            x = x.detach()

        return x

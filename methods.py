import os
import random
from functools import partial
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils_tgr import ROOT_PATH
from dataset import params
from model import get_model
from bsr_transform import BSRTransform

# ============================================================
# Utility – deterministic behaviour
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ============================================================
# Base class (single-model)
# ============================================================

class BaseAttack(object):
    def __init__(self, attack_name, model_name, target):
        self.attack_name = attack_name
        self.model_name = model_name
        self.target = target
        self.loss_flag = -1 if target else 1
        self.used_params = params(self.model_name)

        # load model
        self.model = get_model(self.model_name)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    # --------------------------------------------------------
    # helpers for (un)normalisation
    # --------------------------------------------------------

    def _mul_std_add_mean(self, inps):
        dtype = inps.dtype
        device = inps.device
        mean = torch.as_tensor(self.used_params["mean"], dtype=dtype, device=device)
        std  = torch.as_tensor(self.used_params["std"],  dtype=dtype, device=device)
        inps.mul_(std[:, None, None]).add_(mean[:, None, None])
        return inps

    def _sub_mean_div_std(self, inps):
        dtype = inps.dtype
        device = inps.device
        mean = torch.as_tensor(self.used_params["mean"], dtype=dtype, device=device)
        std  = torch.as_tensor(self.used_params["std"],  dtype=dtype, device=device)
        return (inps - mean[:, None, None]) / std[:, None, None]

    # --------------------------------------------------------
    # I/O helpers
    # --------------------------------------------------------

    def _save_images(self, inps, filenames, output_dir):
        unnorm_inps = self._mul_std_add_mean(inps.clone().detach())
        for i, filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            image = unnorm_inps[i].permute(1, 2, 0)  # C,H,W → H,W,C
            image.clamp_(0, 1)
            image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
            image.save(save_path)

    # --------------------------------------------------------
    # update helpers
    # --------------------------------------------------------

    def _update_perts(self, perts, grad, step_size, epsilon):
        perts.add_(step_size * grad.sign()).clamp_(-epsilon, epsilon)
        return perts

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

# ============================================================
# Transferable Gradient Reweighting (TGR) attack
# NOTE: In practice, this behaves like a mean/std-based
#       Adaptive Token Gradient Regularization (ATGR).
# ============================================================

class TGR(BaseAttack):
    def __init__(
        self,
        model_name,
        sample_num_batches=130,
        steps=10,
        epsilon=16/255,
        target=False,
        decay=1.0,

        # ATGR hyperparameters (kept for compatibility, but values are enforced as fixed below)
        atgr_scale: float = 0.75,         # scaling factor 
        atgr_std_mult: float = 1.1,       # std cutoff multiplier 
        atgr_ratio: float = 0.02,         # outlier ratio 
        atgr_eps: float = 1e-12,
        hard_zero_when_rare: bool = True,

        # Keep original gamma values (module-wise scaling)
        gamma_attn: float = 0.25,
        gamma_qkv: float = 0.75,
        gamma_mlp: float = 0.5,
    ):
        super().__init__("TGR", model_name, target)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = self.epsilon / self.steps
        self.decay = float(decay)
        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = int(sample_num_batches)
        self.max_num_batches = int((224 / 16) ** 2)
        assert self.sample_num_batches <= self.max_num_batches

        # ATGR hyperparameters
        self.atgr_scale = 0.75
        self.atgr_std_mult = 1.1
        self.atgr_ratio = 0.02

        self.atgr_eps = float(atgr_eps)
        self.hard_zero_when_rare = bool(hard_zero_when_rare)

        self.gamma_attn = float(gamma_attn)
        self.gamma_qkv = float(gamma_qkv)
        self.gamma_mlp = float(gamma_mlp)

        # BSR transform
        self.bsr = BSRTransform(num_block=2, num_copies=4, max_angles=0.2)

        # hooks
        self._hook_handles = self.register_atgr_hooks(
            model=self.model,
            model_name=self.model_name,
            atgr_scale=self.atgr_scale,
            atgr_std_mult=self.atgr_std_mult,
            atgr_ratio=self.atgr_ratio,
            atgr_eps=self.atgr_eps,
            hard_zero_when_rare=self.hard_zero_when_rare,
            gamma_attn=self.gamma_attn,
            gamma_qkv=self.gamma_qkv,
            gamma_mlp=self.gamma_mlp,
        )

    @staticmethod
    def _replace_first_grad(grad_input, new_first):
        grad_list = list(grad_input)
        grad_list[0] = new_first
        return tuple(grad_list)

    @staticmethod
    def _adaptive_process(
        g: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        std_factor: float,
        std_mult: float,
        atgr_ratio: float,
        atgr_eps: float,
        hard_zero_when_rare: bool,
    ) -> torch.Tensor:

        thresh = float(std_factor) * float(std_mult) * (std + atgr_eps)
        dev = g - mean
        hi = dev.abs() > thresh

        if hard_zero_when_rare:
            ratio = hi.float().mean()
            if float(ratio) < float(atgr_ratio):
                out = g.clone()
                out[hi] = 0
                return out

        return mean + torch.clamp(dev, -thresh, thresh)

    @classmethod
    def register_atgr_hooks(
        cls,
        model: nn.Module,
        model_name: str,

    
        atgr_scale: float = 0.75,
        atgr_std_mult: float = 1.1,
        atgr_ratio: float = 0.02,
        atgr_eps: float = 1e-12,
        hard_zero_when_rare: bool = True,

        # module scaling
        gamma_attn: float = 0.25,
        gamma_qkv: float = 0.75,
        gamma_mlp: float = 0.5,
    ) -> List[torch.utils.hooks.RemovableHandle]:

        handles: List[torch.utils.hooks.RemovableHandle] = []

        def attn_tgr(module, grad_input, grad_output, gamma):
            if grad_input is None or len(grad_input) == 0 or grad_input[0] is None:
                return None

            # scaling(s) + gamma
            out_grad = grad_input[0] * gamma * atgr_scale

            if model_name in ["vit_base_patch16_224", "visformer_small", "pit_b_224"]:
                B, C, H, W = out_grad.shape
                g = out_grad.reshape(B, C, H * W)
                mean = g.mean(dim=1, keepdim=True)
                std = g.std(dim=1, keepdim=True, unbiased=False)
                g2 = cls._adaptive_process(g, mean, std, 1.5, atgr_std_mult, atgr_ratio, atgr_eps, hard_zero_when_rare)
                out_grad = g2.reshape(B, C, H, W)

            elif model_name in ["cait_s24_224"]:
                B, H, W, C = out_grad.shape
                g = out_grad.reshape(B, H * W, C)
                mean = g.mean(dim=0, keepdim=True)
                std = g.std(dim=0, keepdim=True, unbiased=False)
                g2 = cls._adaptive_process(g, mean, std, 1.5, atgr_std_mult, atgr_ratio, atgr_eps, hard_zero_when_rare)
                out_grad = g2.reshape(B, H, W, C)

            return cls._replace_first_grad(grad_input, out_grad)

        def attn_cait_tgr(module, grad_input, grad_output, gamma):
            if grad_input is None or len(grad_input) == 0 or grad_input[0] is None:
                return None
            out_grad = grad_input[0] * gamma * atgr_scale
            mean = out_grad.mean(dim=0, keepdim=True)
            std = out_grad.std(dim=0, keepdim=True, unbiased=False)
            out_grad = cls._adaptive_process(out_grad, mean, std, 1.5, atgr_std_mult, atgr_ratio, atgr_eps, hard_zero_when_rare)
            return cls._replace_first_grad(grad_input, out_grad)

        def q_tgr(module, grad_input, grad_output, gamma):
            if grad_input is None or len(grad_input) == 0 or grad_input[0] is None:
                return None
            out_grad = grad_input[0] * gamma * atgr_scale
            mean = out_grad.mean(dim=0, keepdim=True)
            std = out_grad.std(dim=0, keepdim=True, unbiased=False)
            out_grad = cls._adaptive_process(out_grad, mean, std, 12.0, atgr_std_mult, atgr_ratio, atgr_eps, hard_zero_when_rare)
            return cls._replace_first_grad(grad_input, out_grad)

        def v_tgr(module, grad_input, grad_output, gamma):
            if grad_input is None or len(grad_input) == 0 or grad_input[0] is None:
                return None
            out_grad = grad_input[0] * gamma * atgr_scale

            if model_name in ["visformer_small"]:
                B, C, H, W = out_grad.shape
                g = out_grad.reshape(B, C, H * W)
                mean = g.mean(dim=1, keepdim=True)
                std = g.std(dim=1, keepdim=True, unbiased=False)
                g2 = cls._adaptive_process(g, mean, std, 12.0, atgr_std_mult, atgr_ratio, atgr_eps, hard_zero_when_rare)
                out_grad = g2.reshape(B, C, H, W)
            else:
                mean = out_grad.mean(dim=0, keepdim=True)
                std = out_grad.std(dim=0, keepdim=True, unbiased=False)
                out_grad = cls._adaptive_process(out_grad, mean, std, 12.0, atgr_std_mult, atgr_ratio, atgr_eps, hard_zero_when_rare)

            return cls._replace_first_grad(grad_input, out_grad)

        def mlp_tgr(module, grad_input, grad_output, gamma):
            if grad_input is None or len(grad_input) == 0 or grad_input[0] is None:
                return None
            out_grad = grad_input[0] * gamma * atgr_scale

            if model_name in ["visformer_small"]:
                B, C, H, W = out_grad.shape
                g = out_grad.reshape(B, C, H * W)
                mean = g.mean(dim=1, keepdim=True)
                std = g.std(dim=1, keepdim=True, unbiased=False)
                g2 = cls._adaptive_process(g, mean, std, 15.0, atgr_std_mult, atgr_ratio, atgr_eps, hard_zero_when_rare)
                out_grad = g2.reshape(B, C, H, W)
            else:
                mean = out_grad.mean(dim=0, keepdim=True)
                std = out_grad.std(dim=0, keepdim=True, unbiased=False)
                out_grad = cls._adaptive_process(out_grad, mean, std, 15.0, atgr_std_mult, atgr_ratio, atgr_eps, hard_zero_when_rare)

            return cls._replace_first_grad(grad_input, out_grad)

        attn_tgr_hook = partial(attn_tgr, gamma=gamma_attn)
        attn_cait_tgr_hook = partial(attn_cait_tgr, gamma=gamma_attn)
        v_tgr_hook = partial(v_tgr, gamma=gamma_qkv)
        q_tgr_hook = partial(q_tgr, gamma=gamma_qkv)
        mlp_tgr_hook = partial(mlp_tgr, gamma=gamma_mlp)

        try:
            if model_name in ["vit_base_patch16_224", "deit_base_distilled_patch16_224"]:
                for i in range(12):
                    handles.append(model.blocks[i].attn.attn_drop.register_full_backward_hook(attn_tgr_hook))
                    handles.append(model.blocks[i].attn.qkv.register_full_backward_hook(v_tgr_hook))
                    handles.append(model.blocks[i].mlp.register_full_backward_hook(mlp_tgr_hook))

            elif model_name == "pit_b_224":
                for block_ind in range(13):
                    if block_ind < 3:
                        t_idx, b_idx = 0, block_ind
                    elif block_ind < 9:
                        t_idx, b_idx = 1, block_ind - 3
                    else:
                        t_idx, b_idx = 2, block_ind - 9
                    blk = model.transformers[t_idx].blocks[b_idx]
                    handles.append(blk.attn.attn_drop.register_full_backward_hook(attn_tgr_hook))
                    handles.append(blk.attn.qkv.register_full_backward_hook(v_tgr_hook))
                    handles.append(blk.mlp.register_full_backward_hook(mlp_tgr_hook))

            elif model_name == "cait_s24_224":
                for block_ind in range(26):
                    if block_ind < 24:
                        blk = model.blocks[block_ind]
                        handles.append(blk.attn.attn_drop.register_full_backward_hook(attn_tgr_hook))
                        handles.append(blk.attn.qkv.register_full_backward_hook(v_tgr_hook))
                        handles.append(blk.mlp.register_full_backward_hook(mlp_tgr_hook))
                    else:
                        blk = model.blocks_token_only[block_ind - 24]
                        handles.append(blk.attn.attn_drop.register_full_backward_hook(attn_cait_tgr_hook))
                        handles.append(blk.attn.q.register_full_backward_hook(q_tgr_hook))
                        handles.append(blk.attn.k.register_full_backward_hook(v_tgr_hook))
                        handles.append(blk.attn.v.register_full_backward_hook(v_tgr_hook))
                        handles.append(blk.mlp.register_full_backward_hook(mlp_tgr_hook))

            elif model_name == "visformer_small":
                for block_ind in range(8):
                    if block_ind < 4:
                        blk = model.stage2[block_ind]
                    else:
                        blk = model.stage3[block_ind - 4]
                    handles.append(blk.attn.attn_drop.register_full_backward_hook(attn_tgr_hook))
                    handles.append(blk.attn.qkv.register_full_backward_hook(v_tgr_hook))
                    handles.append(blk.mlp.register_full_backward_hook(mlp_tgr_hook))

        except Exception as e:
            print(f"[warn] Failed to register ATGR hooks for {model_name}: {e}")

        return handles

    def forward(self, inps, labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inps = inps.to(device)
        labels = labels.to(device)
        criterion = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(inps, device=device)
        unnorm_inps = self._mul_std_add_mean(inps.clone().detach())
        perts = torch.zeros_like(unnorm_inps, device=device).requires_grad_(True)

        k_bsr = 4
        use_bsr = True

        for _ in range(self.steps):
            grads = torch.zeros_like(perts, device=device)
            for _t in range(k_bsr if use_bsr else 1):
                x_t = torch.clamp(unnorm_inps + perts, 0.0, 1.0)
                if use_bsr:
                    x_t = self.bsr._shuffle_rotate(x_t)

                x_t = self._sub_mean_div_std(x_t)
                outputs = self.model(x_t)
                loss = self.loss_flag * criterion(outputs, labels)
                loss.backward()

                grads += perts.grad.data
                perts.grad.data.zero_()

            grads /= (k_bsr if use_bsr else 1)

            grads /= torch.mean(torch.abs(grads), dim=[1, 2, 3], keepdim=True).clamp_min(1e-12)
            grads = grads + momentum * self.decay
            momentum = grads

            perts.data = self._update_perts(perts.data, momentum, self.step_size, self.epsilon)
            perts.data = torch.clamp(unnorm_inps + perts.data, 0.0, 1.0) - unnorm_inps

        adv = self._sub_mean_div_std(torch.clamp(unnorm_inps + perts.data, 0.0, 1.0))
        return adv, None

# ============================================================
# BSR-SAGE 
# = BSR + EGSAM(two-step) + ATGR hook + Momentum 
# ============================================================

class BSR_SAGE(object):
    def __init__(
        self,
        model_name: str,
        steps: int = 10,
        epsilon: float = 16 / 255,
        decay: float = 1.0,
        rho: float = None,
        k_bsr: int = 4,
        num_block: int = 2,
        num_copies: int = 4,
        max_angles: float = 0.2,
        target: bool = False,

        # ATGR hyperparameters
        atgr_scale: float = 0.75,          # scaling factor 
        atgr_std_mult: float = 1.1,        # std cutoff multiplier 
        atgr_ratio: float = 0.02,          # outlier ratio 
        atgr_eps: float = 1e-12,
        hard_zero_when_rare: bool = True,

        # module scaling
        gamma_attn: float = 0.25,
        gamma_qkv: float = 0.75,
        gamma_mlp: float = 0.5,

        surrogate_model_names: Tuple[str, ...] = None,
    ):
        self.attack_name = "BSR_SAGE"
        self.model_name = model_name
        self.target = target
        self.loss_flag = -1 if target else 1

        self.steps = int(steps)
        self.epsilon = float(epsilon)
        self.step_size = self.epsilon / self.steps
        self.decay = float(decay)
        self.k_bsr = int(k_bsr)

        self.rho = float(rho) if rho is not None else float(self.step_size)

        # ATGR hyperparameters
        self.atgr_scale = 0.75
        self.atgr_std_mult = 1.1
        self.atgr_ratio = 0.02

        self.atgr_eps = float(atgr_eps)
        self.hard_zero_when_rare = bool(hard_zero_when_rare)

        self.gamma_attn = float(gamma_attn)
        self.gamma_qkv = float(gamma_qkv)
        self.gamma_mlp = float(gamma_mlp)

        # BSR
        self.bsr = BSRTransform(num_block=num_block, num_copies=num_copies, max_angles=max_angles)

        # surrogate ensemble
        if surrogate_model_names is None:
            if model_name == "vit_base_patch16_224":
                surrogate_model_names = ("vit_base_patch16_224", "visformer_small")
            elif model_name == "visformer_small":
                surrogate_model_names = ("visformer_small", "vit_base_patch16_224")
            else:
                surrogate_model_names = (model_name,)
        self.surrogate_names = tuple(surrogate_model_names)

        # params per surrogate
        self.params_map: Dict[str, Dict] = {name: params(name) for name in self.surrogate_names}

        # reference params for save/IO
        self.ref_name = self.surrogate_names[0]
        self.ref_params = self.params_map[self.ref_name]

        # load surrogate models + ATGR hooks
        self.models: Dict[str, nn.Module] = {}
        self.hook_handles: Dict[str, List[torch.utils.hooks.RemovableHandle]] = {}

        loaded_names: List[str] = []
        for name in self.surrogate_names:
            try:
                m = get_model(name)
                m.requires_grad_(False)
                if torch.cuda.is_available():
                    m = m.cuda()
                m.eval()
                self.models[name] = m
                loaded_names.append(name)

                self.hook_handles[name] = TGR.register_atgr_hooks(
                    model=m,
                    model_name=name,
                    atgr_scale=self.atgr_scale,
                    atgr_std_mult=self.atgr_std_mult,
                    atgr_ratio=self.atgr_ratio,
                    atgr_eps=self.atgr_eps,
                    hard_zero_when_rare=self.hard_zero_when_rare,
                    gamma_attn=self.gamma_attn,
                    gamma_qkv=self.gamma_qkv,
                    gamma_mlp=self.gamma_mlp,
                )
            except Exception as e:
                print(f"[warn] Failed to load surrogate model '{name}': {e}")

        if len(self.models) == 0:
            raise RuntimeError("No surrogate models loaded for BSR_SAGE")

        self.surrogate_names = tuple(loaded_names)
        self.ref_name = self.surrogate_names[0]
        self.ref_params = self.params_map[self.ref_name]

        self.criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # (un)normalize helpers
    # ---------------------------
    def _mul_std_add_mean_ref(self, inps: torch.Tensor) -> torch.Tensor:
        dtype = inps.dtype
        device = inps.device
        mean = torch.as_tensor(self.ref_params["mean"], dtype=dtype, device=device)
        std  = torch.as_tensor(self.ref_params["std"],  dtype=dtype, device=device)
        inps.mul_(std[:, None, None]).add_(mean[:, None, None])
        return inps

    def _sub_mean_div_std_ref(self, inps: torch.Tensor) -> torch.Tensor:
        dtype = inps.dtype
        device = inps.device
        mean = torch.as_tensor(self.ref_params["mean"], dtype=dtype, device=device)
        std  = torch.as_tensor(self.ref_params["std"],  dtype=dtype, device=device)
        return (inps - mean[:, None, None]) / std[:, None, None]

    def _sub_mean_div_std_by_name(self, inps_unnorm: torch.Tensor, name: str) -> torch.Tensor:
        p = self.params_map[name]
        dtype = inps_unnorm.dtype
        device = inps_unnorm.device
        mean = torch.as_tensor(p["mean"], dtype=dtype, device=device)
        std  = torch.as_tensor(p["std"],  dtype=dtype, device=device)
        return (inps_unnorm - mean[:, None, None]) / std[:, None, None]

    # ---------------------------
    # save helper (called by attack runner)
    # ---------------------------
    def _save_images(self, inps, filenames, output_dir):
        unnorm_inps = self._mul_std_add_mean_ref(inps.clone().detach())
        for i, filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            image = unnorm_inps[i].permute(1, 2, 0)
            image.clamp_(0, 1)
            image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
            image.save(save_path)

    # ---------------------------
    # gradient computation with BSR sampling
    # ---------------------------
    def _per_model_grad_with_bsr(
        self,
        unnorm_inps: torch.Tensor,
        perts: torch.Tensor,
        labels: torch.Tensor
    ) -> List[torch.Tensor]:
        grads_list: List[torch.Tensor] = []

        for name in self.surrogate_names:
            model = self.models[name]
            grad_acc = torch.zeros_like(perts)

            for _k in range(self.k_bsr):
                x_t = torch.clamp(unnorm_inps + perts, 0.0, 1.0)
                x_t = self.bsr._shuffle_rotate(x_t)

                x_norm = self._sub_mean_div_std_by_name(x_t, name)
                logits = model(x_norm)
                loss = self.loss_flag * self.criterion(logits, labels)

                grad_t = torch.autograd.grad(
                    loss,
                    perts,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False
                )[0]
                grad_acc += grad_t

            grads_list.append(grad_acc / float(self.k_bsr))

        return grads_list

    def _grad_egsam_step(self, unnorm_inps: torch.Tensor, perts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        per_model_g = self._per_model_grad_with_bsr(unnorm_inps, perts, labels)
        g = torch.stack(per_model_g, dim=0).mean(dim=0)

        with torch.no_grad():
            x_r = torch.clamp(unnorm_inps + perts + self.rho * g.sign(), 0.0, 1.0)
            perts_r = (x_r - unnorm_inps).detach()
        perts_r.requires_grad_(True)

        per_model_gr = self._per_model_grad_with_bsr(unnorm_inps, perts_r, labels)
        g_r = torch.stack(per_model_gr, dim=0).mean(dim=0)

        return g_r

    # ---------------------------
    # main forward
    # ---------------------------
    def forward(self, inps, labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inps = inps.to(device)
        labels = labels.to(device)

        unnorm_inps = self._mul_std_add_mean_ref(inps.clone().detach())
        perts = torch.zeros_like(unnorm_inps, device=device).requires_grad_(True)
        momentum = torch.zeros_like(perts, device=device)

        for _ in range(self.steps):
            g = self._grad_egsam_step(unnorm_inps, perts, labels)

            g = g / torch.mean(torch.abs(g), dim=[1, 2, 3], keepdim=True).clamp_min(self.atgr_eps)
            momentum = momentum * self.decay + g

            with torch.no_grad():
                perts.data = perts.data + self.step_size * momentum.sign()
                perts.data = perts.data.clamp(-self.epsilon, self.epsilon)
                perts.data = torch.clamp(unnorm_inps + perts.data, 0.0, 1.0) - unnorm_inps

        adv_unnorm = torch.clamp(unnorm_inps + perts.data, 0.0, 1.0)
        adv = self._sub_mean_div_std_ref(adv_unnorm)
        return adv, None

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

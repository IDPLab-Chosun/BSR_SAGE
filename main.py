#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attack.py (unified script)
────────────────────────────────────────────────────────────────────────────
Single script supports:
  - Generating adversarial examples (gen)
  - Saving: default .pt tensors (for evaluation consistency; ON by default),
            optional image saving (PNG)
  - Evaluating ASR (eval)
  - Generate + evaluate (gen_eval)

Module priority:
  1) If methods_BSR_SAGE.py exists, load attack classes from there
  2) Otherwise, load from methods.py

Examples)
1) Generate only (save pt + png)
python main.py --mode gen --gpu 0 --model_name vit_base_patch16_224 --attack BSR_SAGE --batch_size 20 --save_png

2) Generate + evaluate
python main.py --mode gen_eval --gpu 0 --model_name vit_base_patch16_224 --attack BSR_SAGE --batch_size 20 --save_png

"""

import os
import argparse
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from timm.models import create_model
from tqdm import tqdm
from PIL import Image  # for PNG saving

# Project modules
from dataset import AdvDataset, params
from utils_tgr import BASE_ADV_PATH, ROOT_PATH


# ─────────────────────────────────────────────────────────────
# 0) Reproducibility
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# 1) Model cache (timm)
# ─────────────────────────────────────────────────────────────
_MODEL_CACHE: Dict[str, torch.nn.Module] = {}


def get_timm_model(model_name: str) -> torch.nn.Module:
    """
    Load a timm pretrained model with caching.
    NOTE: global_pool=None is used to keep behavior consistent with the existing codebase.
    """
    if model_name not in _MODEL_CACHE:
        print(f"[info] Loading model: {model_name}")
        m = create_model(
            model_name,
            pretrained=True,
            num_classes=1000,
            in_chans=3,
            global_pool=None,
        ).eval()
        m.requires_grad_(False)
        if torch.cuda.is_available():
            m = m.cuda()
        _MODEL_CACHE[model_name] = m
    return _MODEL_CACHE[model_name]


# ─────────────────────────────────────────────────────────────
# 2) Load attack module (prefer methods)
# ─────────────────────────────────────────────────────────────
def load_methods_module():
    try:
        import BSR_SAGE.methods as methods_mod  # preferred
        print("[info] Using attacks from: methods.py")
        return methods_mod
    except Exception:
        import methods as methods_mod
        print("[info] Using attacks from: methods.py")
        return methods_mod


# ─────────────────────────────────────────────────────────────
# 3) Saving utility: (A) .pt tensor saving (for evaluation consistency)
# ─────────────────────────────────────────────────────────────
def save_pt_batch(adv: torch.Tensor, names: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for img, name in zip(adv, names):
        stem = Path(name).stem
        torch.save(img.detach().cpu(), out_dir / f"{stem}.pt")


# ─────────────────────────────────────────────────────────────
# 4) Saving utility: (B) PNG saving (for visualization)
#    - methods-style attacks often provide attack_method._save_images which includes
#      the proper inverse-normalization logic.
#    - If not available, fall back to params-based unnormalize and save with PIL.
# ─────────────────────────────────────────────────────────────
MEAN05 = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
STD05 = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)


def unnormalize_by_params(x: torch.Tensor, model_name: str) -> torch.Tensor:
    """Inverse-normalize (normalized tensor) -> (0..1) using dataset.params(model_name) mean/std."""
    p = params(model_name)
    mean = torch.tensor(p["mean"], device=x.device, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(p["std"], device=x.device, dtype=x.dtype).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def unnormalize_05(x: torch.Tensor) -> torch.Tensor:
    """Assuming input is in (-1..1), map to (0..1) using mean=std=0.5."""
    return (x * STD05.to(x.device) + MEAN05.to(x.device)).clamp(0, 1)


def _tensor01_to_pil(img_chw_01: torch.Tensor) -> Image.Image:
    """
    img_chw_01: (C,H,W) in 0..1
    """
    img = img_chw_01.detach().cpu().clamp(0, 1)
    img = (img * 255.0).round().byte()
    img = img.permute(1, 2, 0).numpy()  # HWC
    return Image.fromarray(img)


def save_png_batch(
    adv: torch.Tensor,
    names: List[str],
    out_dir: Path,
    save_mode: str,
    ref_model_name: str,
    attack_method: Optional[Any] = None,
):
    """
    save_mode:
      - "auto": if attack_method._save_images exists, use it (force .png),
                otherwise fall back to params-based unnormalize
      - "params": inverse-normalize using params(mean/std) then save
      - "mean05": inverse-normalize using mean=std=0.5 then save
      - "raw01": assume input is already 0..1 and save as-is
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Use the attack class saving logic if available (better consistency)
    if save_mode == "auto" and attack_method is not None and hasattr(attack_method, "_save_images"):
        forced_names = [f"{Path(n).stem}.png" for n in names]
        attack_method._save_images(adv, forced_names, str(out_dir))
        return

    # 2) Fallback: manual inverse-normalization + PIL saving
    if save_mode == "auto":
        save_mode = "params"

    if save_mode == "params":
        imgs = unnormalize_by_params(adv, ref_model_name)
    elif save_mode == "mean05":
        imgs = unnormalize_05(adv)
    elif save_mode == "raw01":
        imgs = adv.detach().clamp(0, 1)
    else:
        raise ValueError(f"Unknown save_mode: {save_mode}")

    imgs = imgs.detach()
    for im, name in zip(imgs, names):
        stem = Path(name).stem
        path = out_dir / f"{stem}.png"
        pil = _tensor01_to_pil(im)
        pil.save(path, format="PNG")


# ─────────────────────────────────────────────────────────────
# 5) Dataset for evaluation: load adv tensors (.pt) from disk
# ─────────────────────────────────────────────────────────────
class AdvTensorDataset(Dataset):
    def __init__(self, base_dataset: AdvDataset, adv_pt_dir: Path, skip_missing: bool = False):
        self.base_ds = base_dataset
        self.adv_dir = adv_pt_dir
        self.skip_missing = skip_missing

        # If skip_missing=True, exclude samples whose pt file is missing
        self.valid_indices: List[int] = []
        for i in range(len(self.base_ds)):
            _, _, _, name = self.base_ds[i]
            stem = Path(name).stem
            f = self.adv_dir / f"{stem}.pt"
            if f.is_file():
                self.valid_indices.append(i)
            else:
                if not self.skip_missing:
                    self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_idx = self.valid_indices[idx]
        _, label, _, name = self.base_ds[base_idx]
        stem = Path(name).stem
        f = self.adv_dir / f"{stem}.pt"
        if not f.is_file():
            raise FileNotFoundError(f"Missing adversarial tensor: {f}")
        adv = torch.load(f)
        return adv.float(), torch.tensor(label).long()


# ─────────────────────────────────────────────────────────────
# 6) Evaluation: ASR (untargeted)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_asr_one_model(
    model_name: str,
    adv_ds: Dataset,
    batch_size: int,
) -> Tuple[float, int, int]:
    """
    Returns: (ASR%, success_cnt, total_cnt)
    """
    model = get_timm_model(model_name)
    dl = DataLoader(
        adv_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    success = 0
    total = 0
    for adv_imgs, labels in dl:
        if torch.cuda.is_available():
            adv_imgs = adv_imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        logits = model(adv_imgs)

        # Safety: some models may output 4D when global_pool=None
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if logits.ndim > 2:
            logits = F.adaptive_avg_pool2d(logits, 1).squeeze(-1).squeeze(-1)

        preds = logits.argmax(dim=1)
        success += (preds != labels).sum().item()
        total += labels.size(0)

    asr = 100.0 * success / max(total, 1)
    return asr, success, total


# ─────────────────────────────────────────────────────────────
# 7) argparse
# ─────────────────────────────────────────────────────────────
DEFAULT_TEST_MODELS: List[str] = [
    "vit_base_patch16_224",
    "levit_256",
    "cait_s24_224",
    "convit_base",
    "tnt_s_patch16_224",
    "visformer_small",
    "densenet121",
    "inception_v3",
    "vgg16",
    "mobilenetv2_100",
    "resnet152",
    "xception",
    "efficientnet_b3",
]


def parse_args():
    p = argparse.ArgumentParser("Unified attack script (gen/eval/gen_eval)")

    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["gen", "eval", "gen_eval"],
        help="gen: generate, eval: evaluate, gen_eval: generate + evaluate",
    )

    p.add_argument(
        "--attack",
        type=str,
        required=True,
        help="Attack class name defined in methods_BSR_SAGE.py or methods.py",
    )

    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Surrogate/backbone model (timm model name)",
    )

    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Override output root dir (default: BASE_ADV_PATH/<attack_name>)",
    )

    # Saving options
    p.add_argument(
        "--save_pt",
        action="store_true",
        help="(Deprecated) Kept for compatibility. gen saves .pt by default.",
    )
    p.add_argument(
        "--no_save_pt",
        action="store_true",
        help="Disable .pt saving in gen mode (default: ON).",
    )

    # PNG saving option
    p.add_argument(
        "--save_png",
        action="store_true",
        help="Save adversarial images as PNG (for visualization).",
    )

    p.add_argument(
        "--img_mode",
        type=str,
        default="auto",
        choices=["auto", "params", "mean05", "raw01"],
        help="Inverse-normalization strategy for image saving.",
    )

    # Evaluation options
    p.add_argument(
        "--eval_models",
        type=str,
        default="",
        help="Comma-separated evaluation model names (empty => default list).",
    )
    p.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip missing pt files during eval (recommended).",
    )

    # Subdirectories
    p.add_argument(
        "--pt_subdir",
        type=str,
        default="pt",
        help="Subfolder name under out_dir for .pt files.",
    )
    p.add_argument(
        "--png_subdir",
        type=str,
        default="png",
        help="Subfolder name under out_dir for PNG files.",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 8) Generation loop
# ─────────────────────────────────────────────────────────────
def generate_adversarial(args, out_root: Path) -> Tuple[Path, Path]:
    """
    Returns: (pt_dir, png_dir)
    """
    ds_root = Path(ROOT_PATH) / "clean_resized_images"
    dataset = AdvDataset(args.model_name, ds_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    methods_mod = load_methods_module()

    try:
        AttackCls = getattr(methods_mod, args.attack)
    except AttributeError:
        raise ValueError(f"Attack '{args.attack}' not found in methods_BSR_SAGE.py or methods.py")

    # methods-style: AttackCls(model_name)
    attack_method = AttackCls(args.model_name)

    print(f"[info] Attack: {attack_method.__class__.__name__}")
    print(f"[info] Surrogate model_name: {args.model_name}")
    if hasattr(attack_method, "surrogate_names"):
        try:
            print(f"[info] surrogate_names: {getattr(attack_method, 'surrogate_names')}")
        except Exception:
            pass

    pt_dir = out_root / args.pt_subdir
    png_dir = out_root / args.png_subdir

    # In gen mode: save pt by default
    save_pt = (not args.no_save_pt)

    start = time.time()
    for batch_idx, (imgs, labels, _, names) in enumerate(tqdm(loader, desc="[gen] batches")):
        if torch.cuda.is_available():
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        out: Any = attack_method(imgs, labels)
        adv = out[0] if isinstance(out, (list, tuple)) else out

        # Save pt tensors (for evaluation)
        if save_pt:
            save_pt_batch(adv, names, pt_dir)

        # Save PNG images (for visualization)
        if args.save_png:
            save_png_batch(
                adv=adv,
                names=names,
                out_dir=png_dir,
                save_mode=args.img_mode,
                ref_model_name=args.model_name,
                attack_method=attack_method,
            )

        if batch_idx % 100 == 0:
            print(f"[info] gen batch {batch_idx:04d} done")

    print(f"[done] generation finished. elapsed={(time.time() - start) / 60:.1f} min")
    print(f"[out] root: {out_root}")
    if save_pt:
        print(f"[out] pt_dir: {pt_dir}")
    if args.save_png:
        print(f"[out] png_dir: {png_dir}")

    return pt_dir, png_dir


# ─────────────────────────────────────────────────────────────
# 9) Evaluation loop
# ─────────────────────────────────────────────────────────────
def evaluate_from_disk(args, out_root: Path):
    ds_root = Path(ROOT_PATH) / "clean_resized_images"
    base_dataset = AdvDataset(args.model_name, ds_root)

    pt_dir = out_root / args.pt_subdir
    if not pt_dir.exists():
        raise FileNotFoundError(f"pt_dir not found: {pt_dir} (run --mode gen first)")

    adv_ds = AdvTensorDataset(base_dataset, pt_dir, skip_missing=args.skip_missing)

    # Evaluation model list
    if args.eval_models.strip():
        eval_models = [m.strip() for m in args.eval_models.split(",") if m.strip()]
    else:
        eval_models = DEFAULT_TEST_MODELS

    print("\n[eval] ASR evaluation start")
    print(f"[eval] adv_pt_dir: {pt_dir}")
    print(f"[eval] models: {', '.join(eval_models)}")

    results = []
    t0 = time.time()

    for m in eval_models:
        asr, succ, total = evaluate_asr_one_model(m, adv_ds, batch_size=args.batch_size)
        results.append((m, asr, succ, total))
        print(f"[eval] {m:<22s}  ASR={asr:6.2f}%  ({succ}/{total})")

    mean_asr = float(np.mean([r[1] for r in results])) if results else 0.0
    print(f"\n[eval] Mean ASR: {mean_asr:.2f}%")
    print(f"[done] evaluation finished. elapsed={(time.time() - t0) / 60:.1f} min")

    # Save results (csv)
    csv_path = out_root / "asr_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,asr,success,total\n")
        for m, asr, succ, total in results:
            f.write(f"{m},{asr:.4f},{succ},{total}\n")
        f.write(f"MEAN,{mean_asr:.4f},,\n")
    print(f"[save] {csv_path}")


# ─────────────────────────────────────────────────────────────
# 10) main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # GPU visibility (usually takes effect if CUDA context is not initialized yet)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_seed(args.seed)

    safe_attack = Path(args.attack).name  # avoid path-like injection
    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else Path(BASE_ADV_PATH) / f"{safe_attack}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] mode={args.mode}  gpu={args.gpu}  batch_size={args.batch_size}  seed={args.seed}")
    print(f"[info] out_root={out_root}")

    if args.mode == "gen":
        generate_adversarial(args, out_root)

    elif args.mode == "eval":
        evaluate_from_disk(args, out_root)

    elif args.mode == "gen_eval":
        generate_adversarial(args, out_root)
        evaluate_from_disk(args, out_root)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

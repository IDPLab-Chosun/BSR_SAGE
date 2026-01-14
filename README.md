# BSR-SAGE: Sharpness-Aware Gradient Ensemble for Transferable Adversarial Attacks

Official PyTorch implementation for **BSR-SAGE**.

## Requirements
The code works correctly with the following environment:

- **Python** 3.10.19
- **PyTorch** 2.5.1+cu121
- **Torchvision** 0.20.1+cu121
- **Numpy** 2.2.5
- **Pillow** 12.0.0
- **Timm** 0.4.12

## Installation

### 1. Install PyTorch (CUDA 12.1 recommended)
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### 2. Install other dependencies
```bash
pip install timm==0.4.12 numpy pillow tqdm scipy
```

## Experiments

- **ViT models:** Available in the [timm](https://github.com/huggingface/pytorch-image-models) library. We mainly use `vit_base_patch16_224` and `visformer_small` as surrogate models.
- **CNN models:** To evaluate CNN models, please download the converted pretrained models from [tf_to_pytorch_model](https://github.com/ylhz/tf_to_pytorch_model) before running the code. Then place these model checkpoint files in `./models`.

## Data Preparation

Please place your input images in the `clean_resized_images` folder.
Also, ensure that the `image_name_to_class_id_and_name.json` file is present in the root directory (or appropriate path) to map image names to their corresponding class IDs and names.

The directory structure should look like this:
```text
.
├── clean_resized_images/       # Folder containing input images
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── image_name_to_class_id_and_name.json  # Class mapping file
├── main.py
└── ...
```

## Example Usage

You can generate adversarial examples and evaluate them using `main.py`.

### 1) Generate only (Save .pt and .png)
This mode generates adversarial examples and saves them as images and tensors.

```bash
python main.py --mode gen --gpu 0 --model_name vit_base_patch16_224 --attack BSR_SAGE --batch_size 20 --save_png
```

### 2) Generate + Evaluate
This mode generates adversarial examples and immediately evaluates the attack success rate.

```bash
python main.py --mode gen_eval --gpu 0 --model_name vit_base_patch16_224 --attack BSR_SAGE --batch_size 20 --save_png
```

## Acknowledgments
Code refers to:
- [Towards Transferable Adversarial Attacks on Vision Transformers](https://github.com/zhipeng-wei/PNA-PatchOut)
- [Token Gradient Regularization (TGR)](https://github.com/jpzhang1810/TGR)

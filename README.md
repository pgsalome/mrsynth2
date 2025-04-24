# MRSynth2: MRI Sequence Synthesis with GANs

A deep learning repository for synthesizing and translating between MRI sequences using state-of-the-art GAN models.

## Overview

This repository implements multiple image-to-image translation models for MRI sequence synthesis tasks. It enables the generation of one MRI contrast from another (e.g., T1-weighted to T2-weighted) using various GAN architectures. The codebase features a modular and extensible design for neural network research.

## Features

- **Multiple GAN Architectures**:
  - CycleGAN for unpaired image translation
  - Pix2Pix for paired image translation
  - Diffusion models for high-quality generation
  - VAE models for latent space manipulation
  - Latent diffusion models combining VAE and diffusion approaches
  - Self-attention enhanced generators
  - Spectral normalization for discriminator stability

- **Advanced Generators and Discriminators**:
  - ResNet-based generators with 6 or 9 residual blocks
  - U-Net generators of various depths
  - PatchGAN discriminators
  - Multi-scale discriminators for high-resolution images

- **Loss Functions**:
  - Adversarial losses (vanilla, LSGAN)
  - Cycle consistency loss
  - Identity loss
  - Perceptual loss (VGG-based)
  - StyleGAN2-inspired architecture options

- **Optimization**:
  - Bayesian hyperparameter optimization with Optuna
  - Configurable learning rate schedulers (step, cosine, plateau)
  - Extensive metrics for model evaluation (PSNR, SSIM, LPIPS, FID)

- **Experiment Tracking**:
  - Weights & Biases integration
  - TensorBoard logging
  - Comprehensive metrics visualization

- **Complete Preprocessing Pipeline**:
  - DICOM to NIfTI conversion capabilities
  - MRI normalization and bias field correction
  - Registration between different MRI sequences
  - 2D slice extraction with content filtering
  - Paired and unpaired dataset preparation
  - Automatic train/validation/test splitting

- **Reproducibility**:
  - Complete configuration management system
  - Random seed control
  - Result archiving and analysis tools

## Directory Structure

```
mrsynth2/
├── config/                # Configuration files
│   ├── defaults/         # Default configurations
│   │   ├── base.json     # Base configuration
│   │   ├── cyclegan.json # CycleGAN configuration
│   │   ├── diffusion.json # Diffusion model configuration
│   │   ├── pix2pix.json  # Pix2Pix configuration
│   │   ├── vae.json      # VAE configuration
│   │   └── latent_diffusion.json # Latent diffusion configuration
│   └── hyperparameters/  # Hyperparameter search spaces
├── data/                 # Data loading modules
│   ├── datasets/         # Dataset classes
│   │   ├── __init__.py   # Dataset factory
│   │   ├── aligned_dataset.py # Aligned dataset (pix2pix)
│   │   ├── unaligned_dataset.py # Unaligned dataset (cyclegan)
│   │   └── single_dataset.py # Single image dataset (inference)
│   └── dataloader.py     # DataLoader creation utilities
├── models/               # Model implementations
│   ├── __init__.py       # Model factory and registry
│   ├── model_registry.py # Model registration system
│   ├── cycle_gan.py      # CycleGAN implementation
│   ├── pix2pix.py        # Pix2Pix implementation
│   ├── diffusion.py      # Diffusion model implementation
│   ├── vae.py            # VAE model implementation
│   ├── components/       # Shared model components
│   │   ├── __init__.py   # Component exports
│   │   ├── attention.py  # Attention mechanisms
│   │   ├── blocks.py     # Building blocks (ResNet, UNet)
│   │   ├── initialization.py # Weight initialization
│   │   └── normalization.py # Normalization layers
│   ├── generators/       # Generator architectures
│   │   ├── __init__.py   # Generator factory
│   │   ├── resnet_generator.py # ResNet generator
│   │   ├── unet_generator.py # UNet generator
│   │   ├── stylegan_generator.py # StyleGAN2 generator
│   │   └── attention_resnet_generator.py # Attention ResNet
│   └── discriminators/   # Discriminator architectures
│       ├── __init__.py   # Discriminator factory
│       ├── patch_discriminator.py # PatchGAN discriminator
│       ├── spectral_discriminator.py # Spectral norm discriminator
│       └── multiscale_discriminator.py # Multi-scale discriminator
├── scripts/              # Utility scripts
│   ├── evaluate.py       # Evaluation script
│   ├── predict.py        # Inference script
│   ├── run_experiments.py # Experiment runner with Optuna
│   └── train.py          # Training script
├── utils/                # Utility functions
│   ├── cache.py          # Caching utilities
│   ├── config.py         # Configuration utilities 
│   ├── dataclass.py      # Data structures
│   ├── hyperparameter_tuning.py # Hyperparameter tuning
│   ├── image_pool.py     # Image buffer for GANs
│   ├── io.py             # I/O utilities
│   ├── metrics.py        # Evaluation metrics
│   └── perceptual_loss.py # Perceptual losses
├── requirements.txt      # Package dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/mrsynth2.git
cd mrsynth2
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- pandas
- matplotlib
- tqdm
- wandb (for experiment tracking)
- optuna (for hyperparameter optimization)
- scikit-image
- lpips (for perceptual metrics)
- tensorboard
- SimpleITK or nibabel (for medical image processing)
- dicom2nifti (optional, for DICOM conversion)

## Usage

### Configuration

The model and training parameters are controlled through JSON configuration files. Key sections include:

- **data**: Dataset paths, preprocessing, and augmentation
- **model**: Model architecture (generator and discriminator configurations)
- **training**: Training parameters (optimizer, scheduler, loss functions)
- **logging**: Logging and visualization options

Example configuration for CycleGAN:
```json
{
  "data": {
    "dataset_dir": "./data/cyclegan/horse2zebra",
    "batch_size": 8,
    "direction": "AtoB"
  },
  "model": {
    "name": "cyclegan",
    "G_A": {"name": "resnet_9blocks"},
    "G_B": {"name": "resnet_9blocks"},
    "D_A": {"name": "basic"},
    "D_B": {"name": "basic"}
  },
  "training": {
    "loss": {
      "lambda_A": 10.0,
      "lambda_B": 10.0
    }
  }
}
```

### Training

To train a model with the default configuration:

```bash
python scripts/train.py --model_type cyclegan --base_config config/defaults/base.json
```

To customize training parameters:

```bash
python scripts/train.py --model_type cyclegan --base_config config/defaults/base.json --batch_size 4
```

### Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate.py --model_path saved_models/cyclegan_20230415_120000/final_model.pth --config saved_models/cyclegan_20230415_120000/config.json --compute_fid
```

### Inference

To run inference on a single image:

```bash
python scripts/predict.py --model_path saved_models/cyclegan_20230415_120000/final_model.pth --config saved_models/cyclegan_20230415_120000/config.json --input input.png --output_dir results
```

### Hyperparameter Optimization

To run Bayesian hyperparameter optimization with Optuna:

```bash
python scripts/run_experiments.py --base_config config/defaults/base.json --output_dir experiments/optuna --model_type cyclegan --n_trials 50
```

## Extending the Repository

### Adding New Generator Architectures

To add a new generator architecture:
1. Create a new file in `models/generators/`
2. Implement your generator class
3. Register it in `models/generators/__init__.py`

### Adding New Discriminator Architectures

To add a new discriminator architecture:
1. Create a new file in `models/discriminators/`
2. Implement your discriminator class
3. Register it in `models/discriminators/__init__.py`

### Adding New Model Types

To add a new model type:
1. Create a new file in `models/` (e.g., `new_model.py`)
2. Implement your model class
3. Register it in `models/__init__.py`
4. Add default configuration in `config/defaults/new_model.json`

## Weights & Biases Integration

To track experiments with Weights & Biases:

1. Set `logging.wandb.enabled` to `true` in the configuration
2. Configure your project and entity in the configuration or set them up with `wandb login`

```json
"logging": {
  "wandb": {
    "enabled": true,
    "project": "mrsynth2",
    "entity": "your-username",
    "name": "experiment-name",
    "tags": ["cyclegan", "t1-t2"]
  }
}
```

## Citation

If you use this code in your research, please cite:

```
@misc{mrsynth2,
  author = {Author, A.},
  title = {MRSynth2: MRI Sequence Synthesis with GANs},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/username/mrsynth2}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project builds upon methods and architectures from the following works:

- CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (Zhu et al., ICCV 2017)
- Pix2Pix: Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., CVPR 2017)
- Self-Attention GANs (Zhang et al., NIPS 2018)
- StyleGAN2 (Karras et al., CVPR 2020)
- Diffusion Models (Ho et al., NeurIPS 2020)
- Latent Diffusion Models (Rombach et al., CVPR 2022)
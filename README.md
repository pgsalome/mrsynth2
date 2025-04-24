# MRSynth2: MRI Sequence Synthesis with GANs

A deep learning repository for synthesizing and translating between MRI sequences using state-of-the-art GAN models.

## Overview

This repository implements multiple image-to-image translation models for MRI sequence synthesis tasks. It enables the generation of one MRI contrast from another (e.g., T1-weighted to T2-weighted) using various GAN architectures. The codebase features a modular and extensible design based on the structure of the mrclass2 classification repository.

## Features

- **Multiple GAN Architectures**:
  - CycleGAN for unpaired image translation
  - Pix2Pix for paired image translation
  - Diffusion models for high-quality generation
  - VAE models for latent space manipulation
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
  - DICOM to NIfTI conversion
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
│   ├── base.json         # Base configuration
│   ├── cyclegan.json     # CycleGAN configuration
│   ├── diffusion.json    # Diffusion model configuration
│   ├── pix2pix.json      # Pix2Pix configuration
│   ├── vae.json          # VAE configuration
│   └── latent_diffusion  # Latent diffusion configuration
├── preprocessing/        # Preprocessing scripts
│   ├── combine_AB.py     # Combine A and B domains for pix2pix
│   ├── convert_dicom.py  # DICOM to NIfTI conversion
│   ├── create_dataset.py # Dataset creation utilities
│   ├── create_slices.py  # Extract 2D slices from 3D volumes
│   ├── data_preparation.py # Dataset organization and preparation
│   ├── normalize_mri.py  # MRI normalization
│   ├── preprocess.py     # Final image preprocessing and augmentation
│   ├── register_images.py # Registration script
│   └── utils/            # Shared preprocessing utilities
│       ├── registration.py # Registration functions
│       └── slice_extraction.py # Slice extraction functions
├── models/                # Model implementations
│   ├── cycle_gan.py      # CycleGAN implementation
│   ├── discriminator.py  # Discriminator architectures
│   ├── diffusion.py      # Diffusion model implementation
│   ├── generator.py      # Generator architectures
│   ├── model_factory.py  # Factory for creating models
│   ├── pix2pix.py        # Pix2Pix implementation
│   └── vae.py            # VAE model implementation
├── scripts/              # Utility scripts
│   ├── evaluate.py       # Evaluation script
│   ├── predict.py        # Inference script
│   ├── run_experiments.py # Experiment runner with Optuna
│   └── train.py          # Training script
├── src/                  # Source code
│   ├── data_loader.py    # Data loading pipeline
│   └── utils/            # Utility functions
│       ├── cache.py      # Caching utilities
│       ├── config.py     # Configuration utilities 
│       ├── dataclass.py  # Data structures
│       ├── html.py       # HTML visualization
│       ├── image_pool.py # Image buffer for GANs
│       ├── io.py         # I/O utilities
│       ├── metrics.py    # Evaluation metrics
│       ├── perceptual_loss.py # Perceptual losses
│       └── visualizer.py # Visualization utilities
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

### Preprocessing Pipeline

The repository includes a complete preprocessing pipeline for MRI data:

#### 1. DICOM to NIfTI Conversion

Convert DICOM files to NIfTI format:

```bash
python preprocessing/convert_dicom.py --input_dir path/to/dicom --output_dir path/to/nifti
```

#### 2. MRI Normalization

Apply intensity normalization and bias field correction:

```bash
python preprocessing/normalize_mri.py --input_dir path/to/nifti --output_dir path/to/normalized --method z_score
```

#### 3. Registration (for paired data)

Align volumes from different sequences:

```bash
python preprocessing/register_images.py --fixed_dir path/to/t1 --moving_dir path/to/t2 --output_dir path/to/registered
```

#### 4. Slice Extraction

Extract 2D slices from 3D volumes:

```bash
python preprocessing/create_slices.py --input_dir path/to/volumes --output_dir path/to/slices --axis 2
```

#### 5. Dataset Preparation

Prepare datasets for training:

```bash
python preprocessing/data_preparation.py --input_dir path/to/slices --output_dir path/to/dataset --dataset_type cyclegan --domain_a_dir t1 --domain_b_dir t2
```

Or for paired data:

```bash
python preprocessing/combine_AB.py --dir_A path/to/t1 --dir_B path/to/t2 --output_dir path/to/paired
```

#### 6. Final Preprocessing

Apply final preprocessing steps:

```bash
python preprocessing/preprocess.py --config config/base.json --input_dir path/to/dataset --dataset_mode unaligned
```

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
python scripts/train.py --config config/base.json
```

To customize training parameters:

```bash
python scripts/train.py --config config/base.json --batch_size 4 --model_type cyclegan
```

### Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate.py --model_dir saved_models/cyclegan_20230415_120000 --compute_fid
```

### Inference

To run inference on a single image:

```bash
python scripts/predict.py --model_dir saved_models/cyclegan_20230415_120000 --input_image input.png --output_dir results
```

### Hyperparameter Optimization

To run Bayesian hyperparameter optimization with Optuna:

```bash
python scripts/run_experiments.py --base_config config/base.json --output_dir experiments/optuna --n_trials 50 --mode optuna
```

To generate configurations for a grid search:

```bash
python scripts/run_experiments.py --base_config config/base.json --output_dir experiments/grid --mode grid --generate_only
```

## Extending the Repository

### Adding New Generator Architectures

To add a new generator architecture, extend the `models/generator.py` module with your implementation and update the `get_generator` function.

### Adding New Discriminator Architectures

To add a new discriminator architecture, extend the `models/discriminator.py` module with your implementation and update the `get_discriminator` function.

### Adding New Loss Functions

To add a new loss function, implement it in the appropriate model file (e.g., `models/cycle_gan.py`).

### Adding New Preprocessing Steps

To add new preprocessing functionality, add it to the appropriate file in the `preprocessing/` directory or create a new script if needed.

## Examples

### Complete MRI Synthesis Pipeline

```bash
# 1. Convert DICOM to NIfTI
python preprocessing/convert_dicom.py --input_dir data/dicom --output_dir data/nifti

# 2. Normalize MRI data
python preprocessing/normalize_mri.py --input_dir data/nifti --output_dir data/normalized

# 3. Extract slices
python preprocessing/create_slices.py --input_dir data/normalized --output_dir data/slices

# 4. Prepare dataset
python preprocessing/data_preparation.py --input_dir data/slices --output_dir data/prepared --dataset_type cyclegan

# 5. Train CycleGAN model
python scripts/train.py --config config/cyclegan.json --output_dir experiments/t1_to_t2
```

### CycleGAN Training for T1 to T2 Translation

```bash
python scripts/train.py --config config/cyclegan_t1_t2.json
```

### Pix2Pix Training for T1 to FLAIR Translation

```bash
python scripts/train.py --config config/pix2pix_t1_flair.json
```

### Diffusion Model Training

```bash
python scripts/train.py --config config/diffusion.json
```

### Evaluating Multiple Models

```bash
for model in saved_models/*/; do
  python scripts/evaluate.py --model_dir $model --output_dir evaluations/$(basename $model)
done
```

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

The structure is adapted from the mrclass2 repository for MRI classification.
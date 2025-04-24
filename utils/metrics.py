import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Union
import torch.nn.functional as F
import math
from kornia.metrics import ssim as kornia_ssim
from kornia.metrics import psnr as kornia_psnr
import lpips
from skimage.metrics import structural_similarity as ski_ssim
import torchvision.transforms as transforms
from PIL import Image

# Try to import FID score calculation
try:
    from pytorch_fid import fid_score

    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch_fid not available, FID score calculation will be disabled.")


def compute_psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        x: Generated image tensor
        y: Target image tensor
        data_range: Range of the data (default 2.0 for images in [-1, 1])

    Returns:
        PSNR value
    """
    # Ensure same device and correct dimensions
    if x.device != y.device:
        y = y.to(x.device)

    # Use kornia's implementation if available
    try:
        # Kornia's PSNR expects tensors in range [0, 1] or [0, 255]
        # Scale if needed
        if x.min() < 0:
            x_scaled = (x + 1) / 2  # [-1, 1] -> [0, 1]
            y_scaled = (y + 1) / 2
        else:
            x_scaled = x
            y_scaled = y

        return kornia_psnr(x_scaled, y_scaled, max_val=1.0).item()
    except:
        # Fallback to manual calculation
        mse = F.mse_loss(x, y)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(data_range / math.sqrt(mse.item()))


def compute_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> float:
    """
    Compute Structural Similarity Index Measure.

    Args:
        x: Generated image tensor
        y: Target image tensor
        window_size: Size of the gaussian window

    Returns:
        SSIM value
    """
    # Ensure same device and correct dimensions
    if x.device != y.device:
        y = y.to(x.device)

    # Try kornia's SSIM
    try:
        # Kornia's SSIM expects tensors in range [0, 1]
        # Scale if needed
        if x.min() < 0:
            x_scaled = (x + 1) / 2  # [-1, 1] -> [0, 1]
            y_scaled = (y + 1) / 2
        else:
            x_scaled = x
            y_scaled = y

        ssim_val = kornia_ssim(x_scaled, y_scaled, window_size=window_size)
        return ssim_val.mean().item()
    except:
        # Fallback to skimage implementation
        x_np = x.detach().cpu().numpy().transpose(0, 2, 3, 1)
        y_np = y.detach().cpu().numpy().transpose(0, 2, 3, 1)

        if x_np.shape[3] == 1:  # If grayscale, remove channel dimension
            x_np = x_np.squeeze(axis=3)
            y_np = y_np.squeeze(axis=3)

        # Calculate SSIM for each image in the batch
        ssim_vals = []
        for i in range(x_np.shape[0]):
            # Convert to range [0, 1] if needed
            x_i = x_np[i]
            y_i = y_np[i]

            if x_i.min() < 0:
                x_i = (x_i + 1) / 2
                y_i = (y_i + 1) / 2

            ssim_i = ski_ssim(x_i, y_i,
                              data_range=1.0,
                              multichannel=True if x_np.ndim == 4 else False)
            ssim_vals.append(ssim_i)

        return np.mean(ssim_vals)


def compute_lpips(x: torch.Tensor, y: torch.Tensor, lpips_model=None, net: str = 'alex') -> float:
    """
    Compute LPIPS perceptual similarity.

    Args:
        x: Generated image tensor
        y: Target image tensor
        lpips_model: Pre-initialized LPIPS model (optional)
        net: Network to use if lpips_model is None ('alex', 'vgg', 'squeeze')

    Returns:
        LPIPS value (lower means more similar)
    """
    # Initialize LPIPS model if not provided
    if lpips_model is None:
        try:
            lpips_model = lpips.LPIPS(net=net)
            lpips_model.to(x.device)
        except:
            print("LPIPS package not available. Install with: pip install lpips")
            return 0.0

    # Ensure same device
    if x.device != y.device:
        y = y.to(x.device)

    # LPIPS expects input in range [-1, 1]
    # If input is not in this range, normalize
    if x.min() >= 0 and x.max() <= 1:
        x = x * 2 - 1
    if y.min() >= 0 and y.max() <= 1:
        y = y * 2 - 1

    # Compute LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(x, y).mean()

    return lpips_value.item()


def compute_fid(real_images_path: str, generated_images_path: str) -> float:
    """
    Compute Fréchet Inception Distance between real and generated images.

    Args:
        real_images_path: Path to directory with real images
        generated_images_path: Path to directory with generated images

    Returns:
        FID score (lower is better)
    """
    if not FID_AVAILABLE:
        print("FID calculation not available. Install with: pip install pytorch-fid")
        return 0.0

    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [real_images_path, generated_images_path],
            batch_size=50,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dims=2048
        )
        return fid_value
    except Exception as e:
        print(f"Error computing FID score: {e}")
        return 0.0


def compute_kid(real_features: torch.Tensor, gen_features: torch.Tensor,
                subset_size: int = 100, num_subsets: int = 100) -> float:
    """
    Compute Kernel Inception Distance (KID). Uses polynomial kernel with degree 3.

    Args:
        real_features: Features extracted from real images
        gen_features: Features extracted from generated images
        subset_size: Size of the subsets
        num_subsets: Number of subsets to use

    Returns:
        KID score (lower is better)
    """
    n_real = real_features.shape[0]
    n_gen = gen_features.shape[0]

    if n_real < subset_size or n_gen < subset_size:
        print(f"Warning: Not enough samples for KID. Using all {min(n_real, n_gen)} samples.")
        subset_size = min(n_real, n_gen)
        num_subsets = 1

    # Polynomial kernel with degree 3 (cubic)
    def polynomial_kernel(x, y):
        return (torch.mm(x, y.t()) / x.shape[1] + 1) ** 3

    kid_values = []
    for _ in range(num_subsets):
        # Randomly select subsets
        real_idx = np.random.choice(n_real, subset_size, replace=False)
        gen_idx = np.random.choice(n_gen, subset_size, replace=False)

        real_subset = real_features[real_idx]
        gen_subset = gen_features[gen_idx]

        # Compute polynomial kernels
        real_kernel = polynomial_kernel(real_subset, real_subset)
        gen_kernel = polynomial_kernel(gen_subset, gen_subset)
        cross_kernel = polynomial_kernel(real_subset, gen_subset)

        # Compute KID
        kid = real_kernel.mean() + gen_kernel.mean() - 2 * cross_kernel.mean()
        kid_values.append(kid.item())

    return np.mean(kid_values)


def compute_inception_score(probs: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
    """
    Compute Inception Score.

    Args:
        probs: Probabilities from Inception model (softmax outputs)
        splits: Number of splits to compute mean and std

    Returns:
        Tuple of (mean inception score, standard deviation)
    """
    N = probs.shape[0]
    split_size = N // splits

    scores = []
    for k in range(splits):
        part = probs[k * split_size:(k + 1) * split_size]
        py = part.mean(0)
        scores.append(torch.exp(torch.mean(torch.sum(part * (torch.log(part) - torch.log(py)), 1))))

    return torch.mean(torch.stack(scores)).item(), torch.std(torch.stack(scores)).item()


def compute_all_metrics(generated_images: torch.Tensor, target_images: torch.Tensor) -> Dict[str, float]:
    """
    Compute all available metrics between generated and target images.

    Args:
        generated_images: Tensor of generated images
        target_images: Tensor of target images

    Returns:
        Dictionary with metric names and values
    """
    metrics = {}

    # PSNR
    metrics['psnr'] = compute_psnr(generated_images, target_images)

    # SSIM
    metrics['ssim'] = compute_ssim(generated_images, target_images)

    # LPIPS (if available)
    try:
        metrics['lpips'] = compute_lpips(generated_images, target_images)
    except:
        metrics['lpips'] = 0.0

    return metrics


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when a monitored metric has not improved for a given patience.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            mode: 'min' or 'max' depending on whether the monitored metric should be minimized or maximized
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_value: float) -> Tuple[bool, bool]:
        """
        Check if training should be stopped.

        Args:
            metric_value: Current epoch's metric value

        Returns:
            Tuple of (improved, early_stop) booleans
        """
        score = metric_value

        if self.best_score is None:
            self.best_score = score
            return True, False

        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                return True, False
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                return True, False

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
            return False, True

        return False, False
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import torch.nn.functional as F
import math
from PIL import Image


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor: PyTorch tensor

    Returns:
        NumPy array
    """
    # Move tensor to CPU and detach from graph
    tensor = tensor.detach().cpu()

    # Convert to NumPy array
    if tensor.requires_grad:
        tensor = tensor.detach()

    return tensor.numpy()


def ensure_same_dimensions(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure two tensors have the same dimensions.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        Tuple of tensors with same dimensions
    """
    # Check if dimensions match
    if x.shape != y.shape:
        # Resize y to match x
        y = F.interpolate(y, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

    return x, y


def ensure_channel_dimensions(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor has the right channel dimensions.

    Args:
        x: Input tensor

    Returns:
        Tensor with proper channel dimensions
    """
    # Add batch dimension if needed
    if x.dim() == 3:
        x = x.unsqueeze(0)

    # Convert grayscale to RGB if needed
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    return x


def normalize_tensor(x: torch.Tensor, target_range: Tuple[float, float] = (0, 1)) -> torch.Tensor:
    """
    Normalize tensor to target range.

    Args:
        x: Input tensor
        target_range: Target range as (min, max)

    Returns:
        Normalized tensor
    """
    # Get current range
    current_min = x.min()
    current_max = x.max()

    # Handle degenerate case
    if current_min == current_max:
        return torch.ones_like(x) * target_range[0]

    # Normalize to target range
    x_normalized = (x - current_min) / (current_max - current_min)
    x_normalized = x_normalized * (target_range[1] - target_range[0]) + target_range[0]

    return x_normalized


def compute_psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        x: Generated image tensor
        y: Target image tensor
        data_range: Range of the data

    Returns:
        PSNR value
    """
    # Ensure same dimensions and device
    x, y = ensure_same_dimensions(x, y)
    if x.device != y.device:
        y = y.to(x.device)

    # Normalize if needed
    if x.min() < 0 or x.max() > 1:
        x = normalize_tensor(x)
    if y.min() < 0 or y.max() > 1:
        y = normalize_tensor(y)

    # Compute MSE
    mse = F.mse_loss(x, y)

    # Avoid division by zero
    if mse == 0:
        return float('inf')

    # Compute PSNR
    psnr = 20 * math.log10(data_range / math.sqrt(mse.item()))

    return psnr


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
    # Try to use kornia if available
    try:
        import kornia.metrics as metrics

        # Ensure same dimensions and device
        x, y = ensure_same_dimensions(x, y)
        if x.device != y.device:
            y = y.to(x.device)

        # Normalize if needed
        if x.min() < 0 or x.max() > 1:
            x = normalize_tensor(x)
        if y.min() < 0 or y.max() > 1:
            y = normalize_tensor(y)

        # Compute SSIM using kornia
        ssim_val = metrics.ssim(x, y, window_size=window_size)

        return ssim_val.mean().item()

    except ImportError:
        # Fallback to skimage implementation
        try:
            from skimage.metrics import structural_similarity as ski_ssim

            # Convert to numpy arrays
            x_np = tensor_to_numpy(x)
            y_np = tensor_to_numpy(y)

            # Transpose from (B, C, H, W) to (B, H, W, C)
            if x_np.ndim == 4:
                x_np = np.transpose(x_np, (0, 2, 3, 1))
                y_np = np.transpose(y_np, (0, 2, 3, 1))

            # Normalize if needed
            if x_np.min() < 0 or x_np.max() > 1:
                x_np = (x_np - x_np.min()) / (x_np.max() - x_np.min())
            if y_np.min() < 0 or y_np.max() > 1:
                y_np = (y_np - y_np.min()) / (y_np.max() - y_np.min())

            # Handle batch dimension
            if x_np.ndim == 4:
                ssim_vals = []
                for i in range(x_np.shape[0]):
                    ssim_i = ski_ssim(x_np[i], y_np[i], data_range=1.0, multichannel=True)
                    ssim_vals.append(ssim_i)
                return np.mean(ssim_vals)
            else:
                return ski_ssim(x_np, y_np, data_range=1.0, multichannel=(x_np.ndim > 2))

        except ImportError:
            # Basic implementation if neither kornia nor skimage is available
            # Implementation details omitted for brevity
            raise ImportError("Neither kornia nor scikit-image available for SSIM computation")


def compute_lpips(x: torch.Tensor, y: torch.Tensor, net: str = 'alex') -> float:
    """
    Compute LPIPS perceptual similarity.

    Args:
        x: Generated image tensor
        y: Target image tensor
        net: Network to use ('alex', 'vgg', or 'squeeze')

    Returns:
        LPIPS value (lower means more similar)
    """
    try:
        import lpips

        # Ensure proper dimensions and normalization
        x = ensure_channel_dimensions(x)
        y = ensure_channel_dimensions(y)

        # Normalize to [-1, 1] for LPIPS
        if x.min() >= 0 and x.max() <= 1:
            x = x * 2 - 1
        if y.min() >= 0 and y.max() <= 1:
            y = y * 2 - 1

        # Create LPIPS model
        lpips_model = lpips.LPIPS(net=net)
        if x.is_cuda:
            lpips_model = lpips_model.to(x.device)

        # Compute LPIPS
        with torch.no_grad():
            lpips_value = lpips_model(x, y).mean()

        return lpips_value.item()

    except ImportError:
        raise ImportError("LPIPS package not available. Install with: pip install lpips")


def compute_fid(real_path: str, fake_path: str, batch_size: int = 50, dims: int = 2048) -> float:
    """
    Compute Fréchet Inception Distance.

    Args:
        real_path: Path to real images
        fake_path: Path to fake images
        batch_size: Batch size for FID computation
        dims: Dimensionality of Inception features

    Returns:
        FID score (lower is better)
    """
    try:
        from pytorch_fid import fid_score

        # Compute FID
        fid_value = fid_score.calculate_fid_given_paths(
            paths=[real_path, fake_path],
            batch_size=batch_size,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dims=dims
        )

        return fid_value

    except ImportError:
        raise ImportError("pytorch-fid package not available. Install with: pip install pytorch-fid")


class MetricsTracker:
    """Class for tracking and computing multiple metrics."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.batch_sizes = {}

    def update(self, metric_name: str, value: float, batch_size: int = 1) -> None:
        """
        Update a metric.

        Args:
            metric_name: Name of the metric
            value: Value to add
            batch_size: Size of batch for weighted averaging
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = 0.0
            self.batch_sizes[metric_name] = 0

        # Update metric with weighted value
        self.metrics[metric_name] += value * batch_size
        self.batch_sizes[metric_name] += batch_size

    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics.

        Returns:
            Dictionary of metric averages
        """
        return {
            name: value / self.batch_sizes[name]
            for name, value in self.metrics.items()
            if self.batch_sizes[name] > 0
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {}
        self.batch_sizes = {}


class EarlyStopping:
    """Early stopping handler for training."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing metric, 'max' for maximizing
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
            metric_value: Current metric value

        Returns:
            Tuple of (improved, should_stop)
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
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import torchvision.transforms as transforms

from src.utils.io import ensure_dir, save_image
from src.utils.config import read_json_config
from src.utils.metrics import compute_psnr, compute_ssim
from models.model_factory import load_model_from_checkpoint


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Predict with trained image-to-image translation model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model and configuration")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--direction", type=str, choices=["AtoB", "BtoA"],
                        help="Direction for prediction (only for CycleGAN)")
    parser.add_argument("--target_image", type=str, help="Path to target image (optional, for metrics calculation)")
    parser.add_argument("--no_norm", action="store_true", help="Disable input normalization")
    return parser.parse_args()


def load_and_preprocess_image(image_path: str, img_size: int = 256, normalize: bool = True) -> torch.Tensor:
    """
    Load and preprocess an image.

    Args:
        image_path: Path to the image
        img_size: Image size to resize to
        normalize: Whether to normalize the image to [-1, 1]

    Returns:
        Processed image tensor
    """
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Create transform
    transform_list = [
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ]

    # Add normalization if requested
    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform = transforms.Compose(transform_list)

    # Apply transform
    img_tensor = transform(img)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def predict(
        model_dir: str,
        input_image: str,
        output_dir: str = "./results",
        direction: Optional[str] = None,
        target_image: Optional[str] = None,
        no_norm: bool = False
) -> Dict[str, Any]:
    """
    Make a prediction with a trained model.

    Args:
        model_dir: Directory containing the model
        input_image: Path to input image
        output_dir: Directory to save results
        direction: Direction for prediction (only for CycleGAN)
        target_image: Path to target image (optional, for metrics calculation)
        no_norm: Disable input normalization

    Returns:
        Dictionary with results and metrics
    """
    # Set up output directory
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise ValueError(f"Configuration file not found: {config_path}")

    config = read_json_config(str(config_path))
    model_type = config["model"]["name"]

    # Update direction if specified
    if direction:
        config["data"]["direction"] = direction

    # Get image size from config
    img_size = config["data"].get("img_size", 256)

    # Load model using factory
    print(f"Loading {model_type} model from {model_dir}...")
    model = load_model_from_checkpoint(
        model_dir=model_dir,
        epoch="best",
        config=config,
        device=device
    )
    model.eval()

    # Load and preprocess input image
    input_tensor = load_and_preprocess_image(input_image, img_size, normalize=not no_norm)
    input_tensor = input_tensor.to(device)

    # Load target image if provided
    target_tensor = None
    if target_image:
        target_tensor = load_and_preprocess_image(target_image, img_size, normalize=not no_norm)
        target_tensor = target_tensor.to(device)

    # Prepare input data for the model
    data = {
        "A": input_tensor,
        "B": target_tensor if target_tensor is not None else torch.zeros_like(input_tensor)
    }

    # Make prediction
    with torch.no_grad():
        model.set_input(data)
        output = model.test()

    # Get output image
    fake_B = output['fake_B']

    # Calculate metrics if target is provided
    metrics = {}
    if target_tensor is not None:
        print("Calculating metrics...")
        metrics["psnr"] = compute_psnr(fake_B, target_tensor).item()
        metrics["ssim"] = compute_ssim(fake_B, target_tensor).item()
        print(f"PSNR: {metrics['psnr']:.4f}")
        print(f"SSIM: {metrics['ssim']:.4f}")

    # Save images
    input_name = Path(input_image).stem

    # Save input image
    save_image(input_tensor, output_dir / f"{input_name}_input.png", normalize=not no_norm)

    # Save generated image
    save_image(fake_B, output_dir / f"{input_name}_generated.png", normalize=not no_norm)

    # Save target image if provided
    if target_tensor is not None:
        save_image(target_tensor, output_dir / f"{input_name}_target.png", normalize=not no_norm)

    # Save metrics if calculated
    if metrics:
        metrics_path = output_dir / f"{input_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    print(f"Results saved to {output_dir}")

    # Return results
    return {
        "input_path": str(output_dir / f"{input_name}_input.png"),
        "output_path": str(output_dir / f"{input_name}_generated.png"),
        "target_path": str(output_dir / f"{input_name}_target.png") if target_tensor is not None else None,
        "metrics": metrics,
        "model_type": model_type
    }


def main():
    """Main function"""
    args = parse_args()
    results = predict(
        model_dir=args.model_dir,
        input_image=args.input_image,
        output_dir=args.output_dir,
        direction=args.direction,
        target_image=args.target_image,
        no_norm=args.no_norm
    )


if __name__ == "__main__":
    main()
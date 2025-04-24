import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 feature maps to calculate the perceptual
    similarity between generated and target images.
    """

    def __init__(self, feature_layers=[3, 8, 15], weights=[1.0, 1.0, 1.0],
                 use_style_loss=False, style_weight=0.0):
        """
        Initialize perceptual loss.

        Args:
            feature_layers: Feature layers in VGG16 to extract for perceptual loss
            weights: Weights for each feature layer's contribution
            use_style_loss: Whether to include style loss (Gram matrix)
            style_weight: Weight of style loss relative to content loss
        """
        super(PerceptualLoss, self).__init__()

        # Load pre-trained VGG16 and freeze it
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.feature_layers = feature_layers
        self.weights = weights
        self.use_style_loss = use_style_loss
        self.style_weight = style_weight

        # Register forward hooks to get feature maps
        self.outputs = {}
        self.hooks = []

        for i, layer in enumerate(self.vgg):
            if i in self.feature_layers:
                hook = layer.register_forward_hook(self._get_hook(i))
                self.hooks.append(hook)

    def _get_hook(self, idx):
        """Create a hook to store feature maps."""

        def hook(module, input, output):
            self.outputs[idx] = output

        return hook

    def forward(self, gen_img, target_img):
        """
        Calculate perceptual loss between generated and target images.

        Args:
            gen_img: Generated image tensor
            target_img: Target image tensor

        Returns:
            Perceptual loss (and style loss if enabled)
        """
        # Handle grayscale images by converting to 3 channels
        if gen_img.shape[1] == 1:
            gen_img = gen_img.repeat(1, 3, 1, 1)
        if target_img.shape[1] == 1:
            target_img = target_img.repeat(1, 3, 1, 1)

        # Clear feature map outputs
        self.outputs.clear()

        # Get feature maps for generated image
        _ = self.vgg(gen_img)
        gen_features = {k: v for k, v in self.outputs.items()}

        # Clear and get feature maps for target image
        self.outputs.clear()
        _ = self.vgg(target_img)
        target_features = {k: v for k, v in self.outputs.items()}

        # Calculate content loss for each feature map
        content_loss = 0.0
        for i, weight in zip(self.feature_layers, self.weights):
            # L1 loss between feature maps
            content_loss += weight * F.l1_loss(gen_features[i], target_features[i])

        # Calculate style loss (Gram matrix) if enabled
        style_loss = 0.0
        if self.use_style_loss and self.style_weight > 0:
            for i, weight in zip(self.feature_layers, self.weights):
                gen_gram = self._gram_matrix(gen_features[i])
                target_gram = self._gram_matrix(target_features[i])
                style_loss += weight * F.mse_loss(gen_gram, target_gram)

        # Combine losses
        total_loss = content_loss
        if self.use_style_loss:
            total_loss += self.style_weight * style_loss

        return total_loss

    def _gram_matrix(self, x):
        """
        Calculate Gram matrix for style loss.

        The Gram matrix is a matrix of the correlations between feature maps,
        which captures the style of an image.
        """
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        features_t = features.transpose(1, 2)

        # Gram matrix is the correlation between feature maps
        gram = torch.bmm(features, features_t) / (c * h * w)
        return gram

    def __del__(self):
        """Remove hooks when object is deleted."""
        for hook in self.hooks:
            hook.remove()


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) Loss.

    This class requires the lpips package to be installed:
    pip install lpips
    """

    def __init__(self, net='alex', spatial=False, lpips=None):
        """
        Initialize LPIPS loss.

        Args:
            net: Network to use for LPIPS ('alex', 'vgg', or 'squeeze')
            spatial: Whether to return spatial loss map
            lpips: Pre-initialized LPIPS model (optional)
        """
        super(LPIPSLoss, self).__init__()
        try:
            import lpips as lpips_module
            self.lpips_module = lpips_module
        except ImportError:
            raise ImportError('LPIPS loss requires the lpips package. Install with: pip install lpips')

        if lpips is not None:
            self.lpips = lpips
        else:
            self.lpips = self.lpips_module.LPIPS(net=net, spatial=spatial)

    def forward(self, gen_img, target_img):
        """
        Calculate LPIPS perceptual loss.

        Args:
            gen_img: Generated image tensor
            target_img: Target image tensor

        Returns:
            LPIPS loss
        """
        # Handle grayscale images by converting to 3 channels
        if gen_img.shape[1] == 1:
            gen_img = gen_img.repeat(1, 3, 1, 1)
        if target_img.shape[1] == 1:
            target_img = target_img.repeat(1, 3, 1, 1)

        # LPIPS expects input in range [-1, 1]
        # If input is not in this range, normalize
        if gen_img.min() >= 0 and gen_img.max() <= 1:
            gen_img = gen_img * 2 - 1
        if target_img.min() >= 0 and target_img.max() <= 1:
            target_img = target_img * 2 - 1

        return self.lpips(gen_img, target_img).mean()


class VGGStyleLoss(nn.Module):
    """
    Style loss component based on VGG features, useful for style transfer applications.
    """

    def __init__(self, feature_layers=[3, 8, 15, 22], weights=[1.0, 1.0, 1.0, 1.0]):
        """
        Initialize style loss.

        Args:
            feature_layers: Feature layers in VGG to extract
            weights: Weights for each feature layer's contribution
        """
        super(VGGStyleLoss, self).__init__()

        # Load pre-trained VGG16 and freeze it
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.feature_layers = feature_layers
        self.weights = weights

        # Register forward hooks to get feature maps
        self.outputs = {}
        self.hooks = []

        for i, layer in enumerate(self.vgg):
            if i in self.feature_layers:
                hook = layer.register_forward_hook(self._get_hook(i))
                self.hooks.append(hook)

    def _get_hook(self, idx):
        """Create a hook to store feature maps."""

        def hook(module, input, output):
            self.outputs[idx] = output

        return hook

    def forward(self, gen_img, style_img):
        """
        Calculate style loss between generated image and style reference image.

        Args:
            gen_img: Generated image tensor
            style_img: Style reference image tensor

        Returns:
            Style loss
        """
        # Handle grayscale images by converting to 3 channels
        if gen_img.shape[1] == 1:
            gen_img = gen_img.repeat(1, 3, 1, 1)
        if style_img.shape[1] == 1:
            style_img = style_img.repeat(1, 3, 1, 1)

        # Clear feature map outputs
        self.outputs.clear()

        # Get feature maps for generated image
        _ = self.vgg(gen_img)
        gen_features = {k: v for k, v in self.outputs.items()}

        # Clear and get feature maps for style image
        self.outputs.clear()
        _ = self.vgg(style_img)
        style_features = {k: v for k, v in self.outputs.items()}

        # Calculate style loss (Gram matrix)
        style_loss = 0.0
        for i, weight in zip(self.feature_layers, self.weights):
            gen_gram = self._gram_matrix(gen_features[i])
            style_gram = self._gram_matrix(style_features[i])
            style_loss += weight * F.mse_loss(gen_gram, style_gram)

        return style_loss

    def _gram_matrix(self, x):
        """Calculate Gram matrix for style loss."""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        features_t = features.transpose(1, 2)

        # Normalize the Gram matrix
        gram = torch.bmm(features, features_t) / (c * h * w)
        return gram

    def __del__(self):
        """Remove hooks when object is deleted."""
        for hook in self.hooks:
            hook.remove()
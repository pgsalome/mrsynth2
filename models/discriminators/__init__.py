"""Generator factory and exports."""

from typing import Dict, Any

from .resnet_generator import ResnetGenerator
from .unet_generator import UnetGenerator
from .stylegan_generator import StyleGAN2Generator
from .attention_resnet_generator import AttentionResnetGenerator
from ..components.initialization import init_weights


def get_generator(config: Dict[str, Any], input_nc: int, output_nc: int) -> Any:
    """
    Create generator based on configuration.

    Args:
        config: Generator configuration
        input_nc: Number of input channels
        output_nc: Number of output channels

    Returns:
        Generator model
    """
    # Get parameters from config
    name = config.get("name", "resnet_9blocks").lower()
    ngf = config.get("ngf", 64)
    norm_type = config.get("norm", "instance")
    use_dropout = config.get("use_dropout", False)
    init_type = config.get("init_type", "normal")
    init_gain = config.get("init_gain", 0.02)

    # Get appropriate normalization layer
    from ..components.blocks import get_norm_layer
    norm_layer = get_norm_layer(norm_type)

    # Create generator based on type
    if name == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                            use_dropout=use_dropout, n_blocks=9)
    elif name == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                            use_dropout=use_dropout, n_blocks=6)
    elif name == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif name == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif name == 'attention_resnet':
        net = AttentionResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                    use_dropout=use_dropout, n_blocks=9)
    elif name == 'stylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, n_layers=config.get("n_layers", 8),
                                style_dim=config.get("style_dim", 512),
                                use_mapping_network=config.get("use_mapping_network", True))
    else:
        raise NotImplementedError(f'Generator model name [{name}] is not recognized')

    # Initialize weights
    init_weights(net, init_type, init_gain)

    return net
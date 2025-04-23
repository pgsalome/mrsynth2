import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import functools


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block"""
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block"""
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function"""
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations."""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='reflect', use_bias=True, init_type='normal', init_gain=0.02):
        """Initialize the Resnet-based generator"""
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # Resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """U-Net Skip Connection Block"""

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections"""
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """U-Net Generator with skip connections"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, init_type='normal', init_gain=0.02):
        """Construct a Unet generator
        Parameters:
            input_nc (int) -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, if num_downs=7, image of size 128x128 will become of size 1x1 at the bottleneck
            ngf (int) -- the number of filters in the last conv layer
            norm_layer -- normalization layer
        """
        super(UnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        # Construct the Unet structure
        # Add the innermost layer
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)

        # Add intermediate layers with ngf * 8 filters
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)

        # Gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class AttentionResnetGenerator(nn.Module):
    """Resnet-based generator with self-attention layers"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=9, padding_type='reflect', use_attention=True, attention_layers=[5, 6, 7],
                 init_type='normal', init_gain=0.02):
        """Initialize with attention mechanism"""
        super(AttentionResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_attention = use_attention
        self.attention_layers = attention_layers

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # Resnet blocks with attention
        mult = 2 ** n_downsampling
        model_with_attention = []
        for i in range(n_blocks):
            model_with_attention.append(ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                                    use_dropout=use_dropout, use_bias=use_bias))
            # Add self-attention after specified layers
            if use_attention and i in attention_layers:
                model_with_attention.append(SelfAttention(ngf * mult))

        # Upsampling
        upsampling = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            upsampling += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1,
                                              bias=use_bias),
                           norm_layer(int(ngf * mult / 2)),
                           nn.ReLU(True)]
        upsampling += [nn.ReflectionPad2d(3)]
        upsampling += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        upsampling += [nn.Tanh()]

        # Combine the model parts
        self.initial_layers = nn.Sequential(*model)
        self.resnet_attention_blocks = nn.Sequential(*model_with_attention)
        self.upsampling_layers = nn.Sequential(*upsampling)

    def forward(self, input):
        """Forward with attention blocks"""
        out = self.initial_layers(input)
        out = self.resnet_attention_blocks(out)
        out = self.upsampling_layers(out)
        return out


class SelfAttention(nn.Module):
    """Self attention module"""

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()

        # Projections
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)

        # Attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # Output
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Residual connection with learnable parameter
        out = self.gamma * out + x

        return out


# StyleGAN2-based discriminator, for high-quality image synthesis
class StyleGAN2Generator(nn.Module):
    """StyleGAN2-based generator for high-quality image synthesis"""

    def __init__(self, input_nc, output_nc, ngf=64, n_layers=8, style_dim=512,
                 use_mapping_network=True, mapping_layers=8, use_bias=True):
        super(StyleGAN2Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.style_dim = style_dim
        self.use_mapping_network = use_mapping_network

        # Create mapping network if needed
        if use_mapping_network:
            mapping_layers_list = []
            mapping_layers_list.append(nn.Linear(input_nc, style_dim))
            mapping_layers_list.append(nn.LeakyReLU(0.2))

            for i in range(mapping_layers - 1):
                mapping_layers_list.append(nn.Linear(style_dim, style_dim))
                mapping_layers_list.append(nn.LeakyReLU(0.2))

            self.mapping_network = nn.Sequential(*mapping_layers_list)

        # Create synthesis network
        self.synthesis_network = StyleGAN2SynthesisNetwork(
            style_dim=style_dim,
            output_nc=output_nc,
            ngf=ngf,
            n_layers=n_layers,
            use_bias=use_bias
        )

    def forward(self, input):
        if self.use_mapping_network:
            # Process input to extract style vector
            if len(input.shape) == 4:  # If input is an image
                b, c, h, w = input.shape
                input_flat = input.view(b, c * h * w)
                style = self.mapping_network(input_flat)
            else:  # If input is already a vector
                style = self.mapping_network(input)
        else:
            # Treat input directly as style
            style = input

        # Generate image using synthesis network
        output = self.synthesis_network(style)
        return output


class StyleGAN2SynthesisNetwork(nn.Module):
    """StyleGAN2 synthesis network"""

    def __init__(self, style_dim, output_nc, ngf=64, n_layers=8, use_bias=True):
        super(StyleGAN2SynthesisNetwork, self).__init__()

        self.n_layers = n_layers
        self.style_dim = style_dim

        # Initial constant input
        self.constant_input = nn.Parameter(torch.randn(1, ngf, 4, 4))

        # AdaIN layers and convs
        self.style_convs = nn.ModuleList()
        self.adain_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        current_size = 4
        current_channels = ngf

        for i in range(n_layers):
            # Add upsampling for all but first layer
            if i > 0:
                self.upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear'))

            # First conv after upsampling
            self.style_convs.append(
                nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1, bias=use_bias)
            )

            # AdaIN after first conv
            self.adain_layers.append(AdaIN(style_dim, current_channels))

            # Second conv
            next_channels = current_channels // 2 if i > n_layers // 2 else current_channels
            self.style_convs.append(
                nn.Conv2d(current_channels, next_channels, kernel_size=3, padding=1, bias=use_bias)
            )

            # AdaIN after second conv
            self.adain_layers.append(AdaIN(style_dim, next_channels))

            # Update current state
            current_channels = next_channels
            current_size *= 2

        # Final conv to get to output_nc
        self.final_conv = nn.Conv2d(current_channels, output_nc, kernel_size=1, bias=use_bias)
        self.activation = nn.Tanh()

    def forward(self, style):
        # Start from the constant input
        batch_size = style.shape[0]
        x = self.constant_input.repeat(batch_size, 1, 1, 1)

        # Process through the synthesis network
        style_idx = 0
        for i in range(self.n_layers):
            # Upsample if not the first layer
            if i > 0:
                x = self.upsamples[i - 1](x)

            # First conv and AdaIN
            x = self.style_convs[2 * i](x)
            x = self.adain_layers[2 * i](x, style)
            x = F.leaky_relu(x, 0.2)

            # Second conv and AdaIN
            x = self.style_convs[2 * i + 1](x)
            x = self.adain_layers[2 * i + 1](x, style)
            x = F.leaky_relu(x, 0.2)

        # Final processing
        x = self.final_conv(x)
        x = self.activation(x)

        return x


class AdaIN(nn.Module):
    """Adaptive Instance Normalization layer"""

    def __init__(self, style_dim, channel_dim):
        super(AdaIN, self).__init__()

        self.instance_norm = nn.InstanceNorm2d(channel_dim, affine=False)
        self.style_scale = nn.Linear(style_dim, channel_dim)
        self.style_bias = nn.Linear(style_dim, channel_dim)

    def forward(self, x, style):
        # Apply instance norm
        x = self.instance_norm(x)

        # Extract scale and bias from style
        style_scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(style).unsqueeze(2).unsqueeze(3)

        # Apply scale and bias
        return x * style_scale + style_bias


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights."""

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('Initializing network with %s' % init_type)
    net.apply(init_func)


def get_generator(config: Dict[str, Any], input_nc: int, output_nc: int) -> nn.Module:
    """Factory function to create a generator based on configuration"""

    # Get generator type and parameters
    gen_type = config["name"].lower()
    ngf = config.get("ngf", 64)
    norm_type = config.get("norm", "instance")
    use_dropout = config.get("use_dropout", False)
    init_type = config.get("init_type", "normal")
    init_gain = config.get("init_gain", 0.02)

    # Get appropriate normalization layer
    norm_layer = get_norm_layer(norm_type)

    # Create generator based on type
    if gen_type == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                              use_dropout=use_dropout, n_blocks=9)
    elif gen_type == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                              use_dropout=use_dropout, n_blocks=6)
    elif gen_type == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif gen_type == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif gen_type == 'attention_resnet':
        net = AttentionResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                       use_dropout=use_dropout, n_blocks=9)
    elif gen_type == 'stylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, n_layers=config.get("n_layers", 8),
                                 style_dim=config.get("style_dim", 512),
                                 use_mapping_network=config.get("use_mapping_network", True))
    else:
        raise NotImplementedError(f'Generator model name [{gen_type}] is not recognized')

    # Initialize weights
    init_weights(net, init_type, init_gain)

    return net
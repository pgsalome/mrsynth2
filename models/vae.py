import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import numpy as np

from models.generator import ResnetBlock


class VAEEncoder(nn.Module):
    """Encoder network for VAE"""

    def __init__(self, input_nc, output_dim=4, ngf=64, n_downsampling=4, n_blocks=4,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        """
        Initialize VAE Encoder

        Args:
            input_nc: Number of input channels
            output_dim: Dimension of the latent space
            ngf: Number of filters in the last conv layer
            n_downsampling: Number of downsampling operations
            n_blocks: Number of residual blocks after downsampling
            norm_layer: Normalization layer
            padding_type: Type of padding in residual blocks
        """
        super(VAEEncoder, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        # Downsampling
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # Residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=False,
                    use_bias=True
                )
            ]

        # Final convolution to get to latent dimension
        model += [
            nn.Conv2d(ngf * mult, output_dim, kernel_size=3, padding=1, bias=True)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward pass"""
        return self.model(x)


class VAEDecoder(nn.Module):
    """Decoder network for VAE"""

    def __init__(self, input_dim=4, output_nc=3, ngf=64, n_upsampling=4, n_blocks=4,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        """
        Initialize VAE Decoder

        Args:
            input_dim: Dimension of the latent space
            output_nc: Number of output channels
            ngf: Number of filters in the last conv layer
            n_upsampling: Number of upsampling operations
            n_blocks: Number of residual blocks before upsampling
            norm_layer: Normalization layer
            padding_type: Type of padding in residual blocks
        """
        super(VAEDecoder, self).__init__()

        # Initial convolution to map latent to feature space
        model = [
            nn.Conv2d(input_dim, ngf * (2 ** n_upsampling), kernel_size=3, padding=1, bias=True),
            norm_layer(ngf * (2 ** n_upsampling)),
            nn.ReLU(True)
        ]

        # Residual blocks
        mult = 2 ** n_upsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=False,
                    use_bias=True
                )
            ]

        # Upsampling
        for i in range(n_upsampling):
            mult = 2 ** (n_upsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        # Final output convolution
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()  # Output range [-1, 1]
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward pass"""
        return self.model(x)


class VAEModel(nn.Module):
    """VAE Model for Image-to-Image Translation"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VAE model.

        Args:
            config: Configuration dictionary
        """
        super(VAEModel, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.vae_config = self.model_config.get("vae", {})
        self.isTrain = True  # Default to training mode

        # Set up VAE parameters
        self.latent_dim = self.vae_config.get("latent_dim", 4)
        self.kl_weight = self.vae_config.get("kl_weight", 0.0001)
        self.img_size = config["data"].get("img_size", 256)

        # Get input/output channels from config
        self.input_nc = config["data"].get("input_nc", 3)
        self.output_nc = config["data"].get("output_nc", 3)

        # Create encoder and decoder
        encoder_config = self.vae_config.get("encoder", {})
        decoder_config = self.vae_config.get("decoder", {})

        # Standard VAE has separate encoding for mean and log variance
        self.encoder = VAEEncoder(
            input_nc=self.input_nc,
            output_dim=self.latent_dim * 2,  # Both mean and log var
            ngf=encoder_config.get("ngf", 64),
            n_downsampling=encoder_config.get("n_downsampling", 4),
            n_blocks=encoder_config.get("n_blocks", 4),
            norm_layer=nn.BatchNorm2d
        )

        # For conditional VAE, we concatenate the input image to the latent
        if self.vae_config.get("conditional", True):
            # Create condition encoder
            self.cond_encoder = VAEEncoder(
                input_nc=self.input_nc,
                output_dim=self.latent_dim,
                ngf=encoder_config.get("ngf", 64),
                n_downsampling=encoder_config.get("n_downsampling", 4),
                n_blocks=encoder_config.get("n_blocks", 4),
                norm_layer=nn.BatchNorm2d
            )

            # Decoder takes both latent and condition
            self.decoder = VAEDecoder(
                input_dim=self.latent_dim * 2,  # Latent + condition
                output_nc=self.output_nc,
                ngf=decoder_config.get("ngf", 64),
                n_upsampling=decoder_config.get("n_upsampling", 4),
                n_blocks=decoder_config.get("n_blocks", 4),
                norm_layer=nn.BatchNorm2d
            )
        else:
            # Unconditional VAE
            self.decoder = VAEDecoder(
                input_dim=self.latent_dim,
                output_nc=self.output_nc,
                ngf=decoder_config.get("ngf", 64),
                n_upsampling=decoder_config.get("n_upsampling", 4),
                n_blocks=decoder_config.get("n_blocks", 4),
                norm_layer=nn.BatchNorm2d
            )

        # Model names for saving/loading
        if self.vae_config.get("conditional", True):
            self.model_names = ['encoder', 'cond_encoder', 'decoder']
        else:
            self.model_names = ['encoder', 'decoder']

        # Initialize losses and optimizer
        if self.isTrain:
            # Set up losses
            self.loss_names = ['recon', 'kl', 'total']
            self.loss_recon = 0.0
            self.loss_kl = 0.0
            self.loss_total = 0.0
            self._setup_optimizer()

    def _setup_optimizer(self):
        """Set up optimizers"""
        opt_config = self.training_config["optimizer"]

        # Define optimizer parameters
        lr = opt_config.get("lr", 0.0002)
        beta1 = opt_config.get("beta1", 0.5)
        beta2 = opt_config.get("beta2", 0.999)
        weight_decay = opt_config.get("weight_decay", 0.0)

        # Create optimizers
        if self.vae_config.get("conditional", True):
            params = list(self.encoder.parameters()) + list(self.cond_encoder.parameters()) + list(
                self.decoder.parameters())
        else:
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())

        if opt_config["name"].lower() == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
        elif opt_config["name"].lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f'Optimizer {opt_config["name"]} not implemented')

        self.optimizers = [self.optimizer]

    def set_input(self, input):
        """Unpack input data from the dataloader"""
        self.real_A = input['A'].to(self.device)  # input image (condition)
        self.real_B = input['B'].to(self.device)  # target image

    def encode(self, x):
        """Encode image to mean and log variance"""
        h = self.encoder(x)

        # Split into mean and log variance
        h = h.view(h.size(0), 2, self.latent_dim, h.size(2), h.size(3))
        mean, logvar = h[:, 0], h[:, 1]

        return mean, logvar

    def encode_condition(self, x):
        """Encode condition image to latent"""
        if not self.vae_config.get("conditional", True):
            return None

        return self.cond_encoder(x)

    def decode(self, z, cond=None):
        """Decode from latent space to image space"""
        if self.vae_config.get("conditional", True) and cond is not None:
            # Concatenate latent and condition
            z = torch.cat([z, cond], dim=1)

        return self.decoder(z)

    def reparameterize(self, mean, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self):
        """Forward pass"""
        # Encode target image to get mean and log variance
        self.mean, self.logvar = self.encode(self.real_B)

        # Reparameterize to get latent vector
        self.z = self.reparameterize(self.mean, self.logvar)

        # Encode condition if conditional VAE
        if self.vae_config.get("conditional", True):
            self.cond_latent = self.encode_condition(self.real_A)
        else:
            self.cond_latent = None

        # Decode to get reconstructed image
        self.recon_B = self.decode(self.z, self.cond_latent)

    def compute_kl_loss(self):
        """Compute KL divergence loss"""
        # KL divergence from N(mean, var) to N(0, 1)
        kl_loss = -0.5 * torch.sum(1 + self.logvar - self.mean.pow(2) - self.logvar.exp())

        # Normalize by number of elements
        kl_loss = kl_loss / self.mean.numel()

        return kl_loss

    def compute_recon_loss(self):
        """Compute reconstruction loss"""
        # Reconstruction loss (L1 or L2)
        loss_type = self.vae_config.get("recon_loss", "l1")

        if loss_type == "l1":
            recon_loss = F.l1_loss(self.recon_B, self.real_B)
        elif loss_type == "l2":
            recon_loss = F.mse_loss(self.recon_B, self.real_B)
        else:
            raise ValueError(f"Unknown reconstruction loss type: {loss_type}")

        return recon_loss

    def compute_loss(self):
        """Compute VAE loss"""
        # Reconstruction loss
        self.loss_recon = self.compute_recon_loss()

        # KL divergence loss
        self.loss_kl = self.compute_kl_loss()

        # Total loss
        self.loss_total = self.loss_recon + self.kl_weight * self.loss_kl

        return self.loss_total

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # Forward pass
        self.forward()

        # Zero gradients
        self.optimizer.zero_grad()

        # Compute loss
        loss = self.compute_loss()

        # Backward pass
        loss.backward()

        # Update weights
        self.optimizer.step()

    def test(self):
        """Forward function used in test time"""
        with torch.no_grad():
            # Forward pass
            self.forward()

            # For sampling mode, we can directly generate without encoding the target
            if self.vae_config.get("sampling_mode", False):
                # Sample from standard normal
                z = torch.randn(self.real_A.size(0), self.latent_dim,
                                self.real_A.size(2) // (2 ** 4), self.real_A.size(3) // (2 ** 4),
                                device=self.device)

                # Encode condition if conditional VAE
                if self.vae_config.get("conditional", True):
                    cond_latent = self.encode_condition(self.real_A)
                else:
                    cond_latent = None

                # Decode to get generated image
                self.fake_B = self.decode(z, cond_latent)
            else:
                self.fake_B = self.recon_B

            return {'real_A': self.real_A, 'fake_B': self.fake_B, 'real_B': self.real_B}

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = {}
        for name in ['real_A', 'fake_B', 'real_B']:
            if hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return training losses"""
        errors_ret = {}
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk"""
        save_path = self.config["logging"]["save_model_dir"]
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path_model = os.path.join(save_path, save_filename)
                net = getattr(self, name)

                torch.save(net.cpu().state_dict(), save_path_model)
                net.to(self.device)

    def load_networks(self, epoch):
        """Load all the networks from the disk"""
        save_path = self.config["logging"]["save_model_dir"]
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(save_path, load_filename)
                net = getattr(self, name)

                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def update_learning_rate(self, metric=None):
        """Update learning rates for all the networks; called at the end of every epoch"""
        scheduler_config = self.training_config["scheduler"]
        scheduler_name = scheduler_config["name"]

        if scheduler_name == "none":
            return

        # Apply learning rate schedulers
        for scheduler in self.schedulers:
            if scheduler_name == 'plateau':
                scheduler.step(metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def setup_schedulers(self):
        """Set up schedulers"""
        scheduler_config = self.training_config["scheduler"]
        scheduler_name = scheduler_config["name"]

        self.schedulers = []

        if scheduler_name == "none":
            return

        for optimizer in self.optimizers:
            if scheduler_name == 'linear':
                def lambda_rule(epoch):
                    # Linear decay from 1.0 to 0 over n_epochs_decay
                    lr_l = 1.0 - max(0, epoch - self.training_config.get("n_epochs", 100)) / float(
                        self.training_config.get("n_epochs_decay", 100) + 1)
                    return lr_l

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
            elif scheduler_name == 'step':
                step_size = scheduler_config["params"].get("step_size", 30)
                gamma = scheduler_config["params"].get("gamma", 0.1)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_name == 'plateau':
                mode = scheduler_config["params"].get("mode", 'min')
                factor = scheduler_config["params"].get("factor", 0.1)
                patience = scheduler_config["params"].get("patience", 10)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor,
                                                                       patience=patience)
            elif scheduler_name == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max=self.training_config.get("n_epochs", 200))
            else:
                raise NotImplementedError(f'Learning rate scheduler {scheduler_name} is not implemented')

            self.schedulers.append(scheduler)

    def train(self):
        """Set models to training mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()
        self.isTrain = True

    def eval(self):
        """Set models to evaluation mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()
        self.isTrain = False

    @property
    def device(self):
        """Get device for model"""
        return next(self.parameters()).device   
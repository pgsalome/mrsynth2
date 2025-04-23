import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import math
import os
from tqdm import tqdm

from models.generator import UnetGenerator


class DiffusionModel(nn.Module):
    """Diffusion Model for Image-to-Image Translation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the diffusion.json model.

        Args:
            config: Configuration dictionary
        """
        super(DiffusionModel, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.diffusion_config = self.model_config.get("diffusion.json", {})
        self.isTrain = True  # Default to training mode

        # Set up diffusion.json hyperparameters
        self.timesteps = self.diffusion_config.get("timesteps", 1000)
        self.beta_schedule = self.diffusion_config.get("beta_schedule", "linear")
        self.beta_start = self.diffusion_config.get("beta_start", 1e-4)
        self.beta_end = self.diffusion_config.get("beta_end", 2e-2)
        self.img_size = config["data"].get("img_size", 256)

        # Get input/output channels from config
        self.input_nc = config["data"].get("input_nc", 3)
        self.output_nc = config["data"].get("output_nc", 3)

        # Define UNet model for noise prediction
        # The conditional model takes both noisy image and condition as input
        unet_config = self.model_config.get("unet", {})
        unet_type = unet_config.get("name", "unet_256")

        # Define the network for noise prediction
        # For conditional generation, we concatenate the conditioning image with the noisy image
        self.denoise_model = UnetGenerator(
            self.output_nc + self.input_nc,  # Input channels (noisy image + condition)
            self.output_nc,  # Output channels (noise prediction)
            unet_config.get("num_downs", 8),  # Number of downsampling layers
            unet_config.get("ngf", 64),  # Number of filters
            norm_layer=nn.BatchNorm2d,
            use_dropout=unet_config.get("use_dropout", True)
        )

        # Set up diffusion.json process parameters
        self.register_buffer('betas', self._get_beta_schedule())
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

        # Pre-compute diffusion.json process helper values
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1 / self.alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1 / self.alphas_cumprod - 1))

        # Set up posterior parameters (q(x_{t-1} | x_t, x_0))
        self.register_buffer('posterior_variance',
                             self.betas * (1 - self.alphas_cumprod.clone() / self.alphas_cumprod)
                             )
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
                             )
        self.register_buffer('posterior_mean_coef1',
                             torch.sqrt(self.alphas) * (1 - self.alphas_cumprod) / (1 - self.alphas)
                             )
        self.register_buffer('posterior_mean_coef2',
                             torch.sqrt(self.alphas_cumprod.clone()) * self.betas / (1 - self.alphas_cumprod)
                             )

        # Model names for saving/loading
        self.model_names = ['denoise_model']

        # Initialize optimizer
        if self.isTrain:
            # Set up losses
            self.loss_names = ['diffusion.json']
            self.loss_diffusion = 0.0
            self._setup_optimizer()

    def _get_beta_schedule(self) -> torch.Tensor:
        """Create the noise schedule"""
        if self.beta_schedule == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        elif self.beta_schedule == 'quadratic':
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.timesteps) ** 2
        elif self.beta_schedule == 'cosine':
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = self.timesteps + 1
            s = 0.008
            x = torch.linspace(0, self.timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

    def _setup_optimizer(self):
        """Set up optimizers"""
        opt_config = self.training_config["optimizer"]

        # Define optimizer parameters
        lr = opt_config.get("lr", 0.0002)
        beta1 = opt_config.get("beta1", 0.5)
        beta2 = opt_config.get("beta2", 0.999)
        weight_decay = opt_config.get("weight_decay", 0.0)

        # Create optimizers
        if opt_config["name"].lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.denoise_model.parameters(),
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
        elif opt_config["name"].lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.denoise_model.parameters(),
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f'Optimizer {opt_config["name"]} not implemented')

        self.optimizers = [self.optimizer]

    def set_input(self, input):
        """Unpack input data from the dataloader"""
        self.real_A = input['A'].to(self.device)  # input image (condition)
        self.real_B = input['B'].to(self.device)  # target image

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion.json process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        # Sample from q(x_t | x_0) = N(sqrt(alpha_cumprod) * x_0, sqrt(1 - alpha_cumprod) * I)
        x_t = self.sqrt_alphas_cumprod[t] * x_0 + self.sqrt_one_minus_alphas_cumprod[t] * noise

        return x_t, noise

    def predict_noise(self, x_t, t, cond):
        """Predict noise from the denoising U-Net"""
        # Condition the denoising process on the input image A
        # Concatenate the noisy image with the condition along the channel dimension
        x_input = torch.cat([x_t, cond], dim=1)

        # Convert t to embedding to provide timestep information
        t_emb = self._get_timestep_embedding(t, x_t.shape[0])

        # TODO: In a more complete implementation, you'd pass t_emb to the UNet
        # For simplicity, we're ignoring the timestep embedding for now

        # Predict the noise
        return self.denoise_model(x_input)

    def _get_timestep_embedding(self, t, batch_size):
        """Convert timestep tensor to embedding"""
        # Handle both single timestep and batched timesteps
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=self.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0)

        # Repeat if needed to match batch size
        if t.shape[0] == 1 and batch_size > 1:
            t = t.repeat(batch_size)

        # Half of the embedding dimensions use sine, half use cosine
        emb_dim = 128  # Dimension of timestep embedding
        emb = torch.zeros(t.shape[0], emb_dim, device=self.device)

        # Position encoding
        pos_enc_base = 10000
        div_term = torch.exp(torch.arange(0, emb_dim, 2, device=self.device) *
                             -(math.log(pos_enc_base) / emb_dim))

        # Calculate position encodings
        pos_enc = t[:, None].float() * div_term[None, :]
        emb[:, 0::2] = torch.sin(pos_enc)
        emb[:, 1::2] = torch.cos(pos_enc)

        return emb

    def p_mean_variance(self, x_t, t, cond):
        """Compute mean and variance for the posterior p(x_{t-1} | x_t)"""
        # Predict the noise
        pred_noise = self.predict_noise(x_t, t, cond)

        # Compute the posterior mean
        posterior_mean = (
                self.posterior_mean_coef1[t] * x_t +
                self.posterior_mean_coef2[t] * pred_noise
        )

        # Extract variance
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance = self.posterior_log_variance_clipped[t]

        return posterior_mean, posterior_variance, posterior_log_variance, pred_noise

    def p_sample(self, x_t, t, cond):
        """Sample from p(x_{t-1} | x_t)"""
        # Get mean and variance
        mean, variance, log_variance, pred_noise = self.p_mean_variance(x_t, t, cond)

        # No noise at timestep 0
        if t == 0:
            return mean

        # Sample from the posterior
        noise = torch.randn_like(x_t)
        return mean + torch.exp(0.5 * log_variance) * noise

    def sample(self, cond, sample_steps=None):
        """Generate a sample through the reverse diffusion.json process"""
        if sample_steps is None:
            sample_steps = self.timesteps

        # Start from pure noise
        batch_size = cond.shape[0]
        shape = (batch_size, self.output_nc, self.img_size, self.img_size)
        x_t = torch.randn(shape, device=self.device)

        # Reverse diffusion.json process
        for t in tqdm(reversed(range(sample_steps)), desc="Sampling", total=sample_steps):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_tensor, cond)

        return x_t

    def compute_loss(self):
        """Compute the diffusion.json loss"""
        # Select random timesteps for each sample in the batch
        batch_size = self.real_B.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        # Add noise to the target image
        noisy_target, noise = self.q_sample(self.real_B, t)

        # Predict the noise
        pred_noise = self.predict_noise(noisy_target, t, self.real_A)

        # Compute the loss (typically MSE between predicted and actual noise)
        # Optionally weight the loss by signal-to-noise ratio (SNR)
        if self.diffusion_config.get("use_snr_weighting", False):
            # SNR weighting as in the improved DDPM paper
            snr = self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
            weights = snr / snr.max()
            loss = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1, 2, 3])
            loss = (loss * weights).mean()
        else:
            loss = F.mse_loss(pred_noise, noise)

        return loss

    def forward(self):
        """Run forward pass (not used in training)"""
        pass

    def backward(self):
        """Calculate loss and gradients"""
        self.loss_diffusion = self.compute_loss()
        self.loss_diffusion.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # Zero gradients
        self.optimizer.zero_grad()

        # Compute backward pass
        self.backward()

        # Update weights
        self.optimizer.step()

    def test(self):
        """Forward function used in test time"""
        with torch.no_grad():
            # Sample from the model
            self.fake_B = self.sample(self.real_A, sample_steps=self.diffusion_config.get("sample_steps", 50))
            return {'real_A': self.real_A, 'fake_B': self.fake_B, 'real_B': self.real_B}

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = {}
        for name in ['real_A', 'fake_B', 'real_B']:
            if hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses"""
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
        """Get device"""
        return next(self.parameters()).device


class LatentDiffusionModel(DiffusionModel):
    """Latent Diffusion Model for Image-to-Image Translation.

    This implementation uses a pretrained VAE to compress images to latent space
    and runs the diffusion.json process in this latent space.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the latent diffusion.json model.

        Args:
            config: Configuration dictionary
        """
        # Change the channels to the latent dimension
        self.latent_dim = config["model"].get("latent_dim", 4)

        # Temporarily set input/output channels for parent constructor
        input_nc = config["data"].get("input_nc", 3)
        output_nc = config["data"].get("output_nc", 3)
        config["data"]["input_nc"] = self.latent_dim
        config["data"]["output_nc"] = self.latent_dim

        # Initialize the parent class with latent dimensions
        super(LatentDiffusionModel, self).__init__(config)

        # Restore original channels
        config["data"]["input_nc"] = input_nc
        config["data"]["output_nc"] = output_nc
        self.input_nc = input_nc
        self.output_nc = output_nc

        # Load the VAE encoder and decoder
        # In a real implementation, these would be loaded from pretrained models
        from models.vae import VAEEncoder, VAEDecoder

        # Create encoder and decoder
        self.encoder = VAEEncoder(
            input_nc=self.output_nc,
            output_dim=self.latent_dim,
            ngf=64,
            norm_layer=nn.BatchNorm2d
        )

        self.decoder = VAEDecoder(
            input_dim=self.latent_dim,
            output_nc=self.output_nc,
            ngf=64,
            norm_layer=nn.BatchNorm2d
        )

        # Add to model names
        self.model_names.extend(['encoder', 'decoder'])

        # Freeze VAE if specified
        if config["model"].get("freeze_vae", True):
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

    def encode(self, x):
        """Encode image to latent space"""
        return self.encoder(x)

    def decode(self, z):
        """Decode from latent space to image space"""
        return self.decoder(z)

    def set_input(self, input):
        """Unpack input data and encode to latent space"""
        self.real_A = input['A'].to(self.device)  # input image (condition)
        self.real_B = input['B'].to(self.device)  # target image

        # Encode condition to latent space (if using latent conditioning)
        if self.diffusion_config.get("latent_conditioning", False):
            with torch.no_grad():
                self.latent_A = self.encode(self.real_A)
        else:
            self.latent_A = self.real_A  # Use pixel-space conditioning

        # Encode target to latent space
        with torch.no_grad():
            self.latent_B = self.encode(self.real_B)

    def compute_loss(self):
        """Compute the diffusion.json loss in latent space"""
        # Select random timesteps for each sample in the batch
        batch_size = self.latent_B.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)

        # Add noise to the latent target
        noisy_latent, noise = self.q_sample(self.latent_B, t)

        # Predict the noise using conditioning
        pred_noise = self.predict_noise(noisy_latent, t,
                                        self.latent_A if self.diffusion_config.get("latent_conditioning",
                                                                                   False) else self.real_A)

        # Compute the loss
        if self.diffusion_config.get("use_snr_weighting", False):
            # SNR weighting
            snr = self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
            weights = snr / snr.max()
            loss = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1, 2, 3])
            loss = (loss * weights).mean()
        else:
            loss = F.mse_loss(pred_noise, noise)

        return loss

    def test(self):
        """Forward function used in test time"""
        with torch.no_grad():
            # Sample latent through the diffusion.json process
            latent_cond = self.latent_A if self.diffusion_config.get("latent_conditioning", False) else self.real_A
            sampled_latent = self.sample(latent_cond, sample_steps=self.diffusion_config.get("sample_steps", 50))

            # Decode the latent to image space
            self.fake_B = self.decode(sampled_latent)

            return {'real_A': self.real_A, 'fake_B': self.fake_B, 'real_B': self.real_B}
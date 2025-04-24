import torch
import torch.nn as nn
from typing import Dict, Any
import os

from models.generator import get_generator
from models.discriminator import get_discriminator
from utils.perceptual_loss import PerceptualLoss


class Pix2PixModel(nn.Module):
    """Pix2Pix Model for paired image-to-image translation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Pix2Pix class.
        Parameters:
            config (Dict[str, Any]): Configuration dictionary
        """
        super(Pix2PixModel, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.isTrain = True  # Default to training mode
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        # Specify input and output channels
        input_nc = config["data"]["input_nc"]
        output_nc = config["data"]["output_nc"]

        # Define networks
        # Generator: G maps from input to output domain
        self.netG = get_generator(self.model_config["G_A"] if "G_A" in self.model_config else self.model_config,
                                  input_nc, output_nc)

        self.model_names = ['G']

        # Initialize discriminator if in training mode
        if self.isTrain:
            # Discriminator: D discriminates real/fake pairs
            self.netD = get_discriminator(self.model_config["D_A"] if "D_A" in self.model_config else self.model_config,
                                          input_nc + output_nc)  # discriminator takes concatenated input
            self.model_names.append('D')

            # Define loss functions
            self.criterionGAN = self._get_gan_loss()
            self.criterionL1 = nn.L1Loss()

            # Perceptual loss (VGG-based) if enabled
            self.use_perceptual_loss = self.training_config["loss"].get("lambda_perceptual", 0.0) > 0
            if self.use_perceptual_loss:
                self.criterionPerceptual = PerceptualLoss()
                self.loss_names.append('G_perceptual')

            # Initialize optimizers
            self._setup_optimizers()

            # Initialize losses dictionary
            self.loss_G_GAN = 0
            self.loss_G_L1 = 0
            self.loss_D_real = 0
            self.loss_D_fake = 0
            if self.use_perceptual_loss:
                self.loss_G_perceptual = 0

    def _get_gan_loss(self):
        """Define GAN loss based on configuration"""
        gan_mode = self.training_config["loss"].get("gan_mode", "vanilla")

        if gan_mode == 'lsgan':
            return nn.MSELoss()
        elif gan_mode == 'vanilla':
            return nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            return lambda prediction, target: -torch.mean(prediction) if target else torch.mean(prediction)
        else:
            raise NotImplementedError(f'GAN loss type {gan_mode} not implemented')

    def _setup_optimizers(self):
        """Set up optimizers for generator and discriminator"""
        opt_config = self.training_config["optimizer"]

        # Define optimizer parameters
        lr = opt_config.get("lr", 0.0002)
        beta1 = opt_config.get("beta1", 0.5)
        beta2 = opt_config.get("beta2", 0.999)
        weight_decay = opt_config.get("weight_decay", 0.0)

        # Create optimizers
        if opt_config["name"].lower() == "adam":
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
        elif opt_config["name"].lower() == "adamw":
            self.optimizer_G = torch.optim.AdamW(
                self.netG.parameters(),
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
            self.optimizer_D = torch.optim.AdamW(
                self.netD.parameters(),
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f'Optimizer {opt_config["name"]} not implemented')

        self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing"""
        self.real_A = input['A'].to(self.device)  # source (input) image
        self.real_B = input['B'].to(self.device)  # target (output) image
        if 'mask' in input:
            self.mask = input['mask'].to(self.device)  # optional mask

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>"""
        # G(A) -> B
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake pairs
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # Concatenate input and generated output
        pred_fake = self.netD(fake_AB.detach())  # Detach to avoid backprop to generator
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real pairs
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

        return self.loss_D

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        lambda_L1 = self.training_config["loss"].get("lambda_L1", 100.0)
        lambda_perceptual = self.training_config["loss"].get("lambda_perceptual", 0.0)

        # First, G(A) should fool the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * lambda_L1

        # Optional perceptual loss
        if lambda_perceptual > 0 and self.use_perceptual_loss:
            self.loss_G_perceptual = self.criterionPerceptual(self.fake_B, self.real_B) * lambda_perceptual
        else:
            self.loss_G_perceptual = 0

        # Compute total generator loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        if self.use_perceptual_loss:
            self.loss_G += self.loss_G_perceptual

        self.loss_G.backward()

        return self.loss_G

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # Forward pass
        self.forward()

        # Update Discriminator
        self.set_requires_grad(self.netD, True)  # Enable backprop for D
        self.optimizer_D.zero_grad()  # Set D's gradients to zero
        self.backward_D()  # Calculate gradients for D
        self.optimizer_D.step()  # Update D's weights

        # Update Generator
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # Set G's gradients to zero
        self.backward_G()  # Calculate graidents for G
        self.optimizer_G.step()  # Update G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def get_current_losses(self):
        """Return traning losses / errors"""
        errors_ret = {}
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

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

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_path = self.config["logging"]["save_model_dir"]
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path_model = os.path.join(save_path, save_filename)
                net = getattr(self, 'net' + name)

                torch.save(net.cpu().state_dict(), save_path_model)
                net.to(self.device)

    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_path = self.config["logging"]["save_model_dir"]
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(save_path, load_filename)
                net = getattr(self, 'net' + name)

                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        """
        with torch.no_grad():
            self.forward()
            return {
                'real_A': self.real_A,
                'fake_B': self.fake_B,
                'real_B': self.real_B
            }

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = {}
        for name in ['real_A', 'fake_B', 'real_B']:
            if hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def validate(self, val_dataset, metrics=['psnr', 'ssim']):
        """Validate model on the validation set
        Parameters:
            val_dataset -- validation dataset
            metrics -- list of metrics to compute
        Returns:
            validation_metrics -- dictionary of validation metrics
        """
        self.eval()  # Set to evaluation mode

        # Initialize metrics
        validation_metrics = {
            'val_l1': 0.0,
            'val_gan': 0.0,
            'val_psnr': 0.0,
            'val_ssim': 0.0
        }

        n_samples = 0

        with torch.no_grad():
            for i, data in enumerate(val_dataset):
                self.set_input(data)
                self.forward()

                # Compute L1 loss
                l1_loss = self.criterionL1(self.fake_B, self.real_B).item()
                validation_metrics['val_l1'] += l1_loss

                # Compute GAN loss
                fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                pred_fake = self.netD(fake_AB)
                gan_loss = self.criterionGAN(pred_fake, True).item()
                validation_metrics['val_gan'] += gan_loss

                # Compute PSNR
                if 'psnr' in metrics:
                    from utils.metrics import compute_psnr
                    psnr = compute_psnr(self.fake_B, self.real_B)
                    validation_metrics['val_psnr'] += psnr

                # Compute SSIM
                if 'ssim' in metrics:
                    from utils.metrics import compute_ssim
                    ssim = compute_ssim(self.fake_B, self.real_B)
                    validation_metrics['val_ssim'] += ssim

                n_samples += 1

        # Average metrics
        for key in validation_metrics:
            validation_metrics[key] /= n_samples

        self.train()  # Set back to training mode

        return validation_metrics

    @property
    def device(self):
        """Get device for model"""
        return next(self.parameters()).device

    def train(self):
        """Set model to training mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()
        self.isTrain = True

    def eval(self):
        """Set model to evaluation mode"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
        self.isTrain = False
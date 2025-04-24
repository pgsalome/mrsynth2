import torch
import torch.nn as nn
import itertools
from typing import Dict, Any

from models.generator import get_generator
from models.discriminator import get_discriminator
from utils.image_pool import ImagePool
from utils.perceptual_loss import PerceptualLoss


class CycleGANModel(nn.Module):
    """CycleGAN Model for unpaired image-to-image translation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the CycleGAN class.
        Parameters:
            config (Dict[str, Any]): Configuration dictionary
        """
        super(CycleGANModel, self).__init__()
        self.config = config
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.isTrain = True  # Default to training mode
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # Specify input and output channels
        input_nc = config["data"]["input_nc"]
        output_nc = config["data"]["output_nc"]

        # Define networks
        # Generators: G_A: A -> B; G_B: B -> A
        self.netG_A = get_generator(self.model_config["G_A"], input_nc, output_nc)
        self.netG_B = get_generator(self.model_config["G_B"], output_nc, input_nc)

        self.model_names = ['G_A', 'G_B']

        # Initialize discriminators if in training mode
        if self.isTrain:
            # Discriminators: D_A discriminates real/fake A; D_B discriminates real/fake B
            self.netD_A = get_discriminator(self.model_config["D_A"], input_nc)
            self.netD_B = get_discriminator(self.model_config["D_B"], output_nc)
            self.model_names.extend(['D_A', 'D_B'])

            # Define image pools for storing previously generated images
            self.fake_A_pool = ImagePool(self.training_config.get("pool_size", 50))
            self.fake_B_pool = ImagePool(self.training_config.get("pool_size", 50))

            # Define loss functions
            self.criterionGAN = self._get_gan_loss()
            self.criterionCycle = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()

            # Perceptual loss (VGG-based) if enabled
            self.use_perceptual_loss = self.training_config["loss"].get("lambda_perceptual", 0.0) > 0
            if self.use_perceptual_loss:
                self.criterionPerceptual = PerceptualLoss()

            # Initialize optimizers
            self._setup_optimizers()

            # Initialize losses dictionary
            self.loss_D_A = 0
            self.loss_D_B = 0
            self.loss_G_A = 0
            self.loss_G_B = 0
            self.loss_cycle_A = 0
            self.loss_cycle_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            if self.use_perceptual_loss:
                self.loss_perceptual_A = 0
                self.loss_perceptual_B = 0
                self.loss_names.extend(['perceptual_A', 'perceptual_B'])

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
        """Set up optimizers for generators and discriminators"""
        opt_config = self.training_config["optimizer"]

        # Define optimizer parameters
        lr = opt_config.get("lr", 0.0002)
        beta1 = opt_config.get("beta1", 0.5)
        beta2 = opt_config.get("beta2", 0.999)
        weight_decay = opt_config.get("weight_decay", 0.0)

        # Create optimizers
        if opt_config["name"].lower() == "adam":
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
        elif opt_config["name"].lower() == "adamw":
            self.optimizer_G = torch.optim.AdamW(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
            self.optimizer_D = torch.optim.AdamW(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
        else:
            raise NotImplementedError(f'Optimizer {opt_config["name"]} not implemented')

        self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing"""
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>"""
        # G_A(A) -> B
        self.fake_B = self.netG_A(self.real_A)
        # G_B(B) -> A
        self.fake_A = self.netG_B(self.real_B)
        # Identity mappings
        self.idt_A = self.netG_A(self.real_B)
        self.idt_B = self.netG_B(self.real_A)
        # Cycle consistency
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A)) -> A
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B)) -> B

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.training_config["loss"].get("lambda_A", 10.0)
        lambda_B = self.training_config["loss"].get("lambda_B", 10.0)
        lambda_idt = self.training_config["loss"].get("lambda_identity", 0.5)
        lambda_perceptual = self.training_config["loss"].get("lambda_perceptual", 0.0)

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_B(B))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_A), True)
        # GAN loss D_B(G_A(A))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_B), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Perceptual loss
        if lambda_perceptual > 0 and self.use_perceptual_loss:
            self.loss_perceptual_A = self.criterionPerceptual(self.fake_A, self.real_A) * lambda_perceptual
            self.loss_perceptual_B = self.criterionPerceptual(self.fake_B, self.real_B) * lambda_perceptual
        else:
            self.loss_perceptual_A = 0
            self.loss_perceptual_B = 0

        # Combined loss and calculate gradients
        total_loss = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        if self.use_perceptual_loss:
            total_loss += self.loss_perceptual_A + self.loss_perceptual_B

        total_loss.backward()

        return total_loss

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # Forward pass
        self.forward()

        # G_A and G_B
        # Disable backprop for discriminators
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()  # Set G's gradients to zero
        self.loss_G = self.backward_G()  # Calculate gradients for G
        self.optimizer_G.step()  # Update G's weights

        # D_A and D_B
        # Enable backprop for discriminators
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # Set D's gradients to zero
        self.backward_D_A()  # Calculate gradients for D_A
        self.backward_D_B()  # Calculate gradients for D_B
        self.optimizer_D.step()  # Update D's weights

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
                'real_B': self.real_B,
                'fake_A': self.fake_A
            }

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = {}
        for name in ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']:
            if hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        return visual_ret

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
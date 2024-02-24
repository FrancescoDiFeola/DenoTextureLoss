import itertools
from models.unit_networks import *
import torch
from collections import OrderedDict
import os
import sys



def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


class UNITModel:

    def __init__(self, opt):
        """Initialize the UNIT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixel = torch.nn.L1Loss()
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.Tensor = torch.FloatTensor if str(self.device).find("cpu") != -1 else torch.Tensor

        # Dimensionality (channel-wise) of image embedding
        shared_dim = opt.dim * 2 ** opt.n_downsample

        # Initialize generator and discriminator
        shared_E = ResidualBlock(features=shared_dim)
        self.E1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
        self.E2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
        shared_G = ResidualBlock(features=shared_dim)
        self.G1 = Generator(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
        self.G2 = Generator(dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
        self.D1 = Discriminator(input_shape)
        self.D2 = Discriminator(input_shape)

        self.E1 = self.E1.to(self.device)
        self.E2 = self.E2.to(self.device)
        self.G1 = self.G1.to(self.device)
        self.G2 = self.G2.to(self.device)
        self.D1 = self.D1.to(self.device)
        self.D2 = self.D2.to(self.device)
        self.criterion_GAN.to(self.device)
        self.criterion_pixel.to(self.device)

        # Initialize weights
        self.E1.apply(weights_init_normal)
        self.E2.apply(weights_init_normal)
        self.G1.apply(weights_init_normal)
        self.G2.apply(weights_init_normal)
        self.D1.apply(weights_init_normal)
        self.D2.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.E1.parameters(), self.E2.parameters(), self.G1.parameters(), self.G2.parameters()),
            lr=opt.lr,
            betas=(opt.b1, opt.b2),
        )
        self.optimizer_D1 = torch.optim.Adam(self.D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D2 = torch.optim.Adam(self.D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        # Learning rate update schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )
        self.lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )
        self.lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
        )

        self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2']
        # dictionary to store training loss
        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        self.opt = opt

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        if self.opt.dataset_mode == "LIDC_IDRI":
            self.X1 = input['img'].type(self.Tensor).expand(-1, 3, -1, -1).to(self.device)
            self.image_paths = input['im_paths']
        else:
            # Set model input
            self.X1 = input["A"].type(self.Tensor).expand(-1, 3, -1, -1).to(self.device)
            self.X2 = input["B"].type(self.Tensor).expand(-1, 3, -1, -1).to(self.device)
            self.image_paths = input['A_paths']

            # Adversarial ground truths
            self.valid = Variable(self.Tensor(np.ones((self.X1.size(0), *self.D1.output_shape))), requires_grad=False)
            self.fake = Variable(self.Tensor(np.zeros((self.X1.size(0), *self.D1.output_shape))), requires_grad=False)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.optimizer_G.zero_grad()

        # Get shared latent representation
        self.mu1, self.Z1 = self.E1(self.X1)
        self.mu2, self.Z2 = self.E2(self.X2)

        # Reconstruct images
        self.recon_X1 = self.G1(self.Z1)
        self.recon_X2 = self.G2(self.Z2)

        # Translate images
        self.fake_X1 = self.G1(self.Z2)
        self.fake_X2 = self.G2(self.Z1)

        # Cycle translation
        self.mu1_, self.Z1_ = self.E1(self.fake_X1)
        self.mu2_, self.Z2_ = self.E2(self.fake_X2)
        self.cycle_X1 = self.G1(self.Z2_)
        self.cycle_X2 = self.G2(self.Z1_)

    def backwark_encoderGen(self):
        self.loss_GAN_1 = self.opt.lambda_0 * self.criterion_GAN(self.D1(self.fake_X1), self.valid)
        self.loss_GAN_2 = self.opt.lambda_0 * self.criterion_GAN(self.D2(self.fake_X2), self.valid)
        self.loss_KL_1 = self.opt.lambda_1 * compute_kl(self.mu1)
        self.loss_KL_2 = self.opt.lambda_1 * compute_kl(self.mu2)
        self.loss_ID_1 = self.opt.lambda_2 * self.criterion_pixel(self.recon_X1, self.X1)
        self.loss_ID_2 = self.opt.lambda_2 * self.criterion_pixel(self.recon_X2, self.X2)
        self.loss_KL_1_ = self.opt.lambda_3 * compute_kl(self.mu1_)
        self.loss_KL_2_ = self.opt.lambda_3 * compute_kl(self.mu2_)
        self.loss_cyc_1 = self.opt.lambda_4 * self.criterion_pixel(self.cycle_X1, self.X1)
        self.loss_cyc_2 = self.opt.lambda_4 * self.criterion_pixel(self.cycle_X2, self.X2)

        # Total loss
        self.loss_G = (
                self.loss_KL_1
                + self.loss_KL_2
                + self.loss_ID_1
                + self.loss_ID_2
                + self.loss_GAN_1
                + self.loss_GAN_2
                + self.loss_KL_1_
                + self.loss_KL_2_
                + self.loss_cyc_1
                + self.loss_cyc_2
        )
        self.loss_G.backward()

    def backward_D1(self):
        self.loss_D1 = self.criterion_GAN(self.D1(self.X1), self.valid) + self.criterion_GAN(self.D1(self.fake_X1.detach()), self.fake)
        self.loss_D1.backward()

    def backward_D2(self):
        self.loss_D2 = self.criterion_GAN(self.D2(self.X2), self.valid) + self.criterion_GAN(self.D2(self.fake_X2.detach()), self.fake)
        self.loss_D2.backward()

    def optimize_parameters(self):
        self.forward()

        self.backwark_encoderGen()
        self.optimizer_G.step()

        self.optimizer_D1.zero_grad()
        self.backward_D1()
        self.optimizer_D1.step()

        self.optimizer_D2.zero_grad()
        self.backward_D2()
        self.optimizer_D2.step()

    def update_learning_rate(self):
        # Update learning rates
        self.lr_scheduler_G.step()
        self.lr_scheduler_D1.step()
        self.lr_scheduler_D2.step()

    #############################################

    def get_current_losses(self):
        """Return traning losses / errors"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def track_current_losses(self):
        """Return traning losses / errors so far."""
        for name in self.loss_names:
            if isinstance(name, str):
                self.error_store[name].append(
                    float(getattr(self, 'loss_' + name)))  # float(...) works for both scalar tensor and float number

        return self.error_store

    def print_current_loss(self, epoch, i, dataset_len, time_left):
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, self.opt.n_epochs, dataset_len, i, (self.loss_D1 + self.loss_D2).item(), self.loss_G.item(), time_left)
        )

    def save_networks(self, epoch):
        torch.save(self.E1.state_dict(), f"{self.save_dir}/E1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.E2.state_dict(), f"{self.save_dir}/E2_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.G1.state_dict(), f"{self.save_dir}/G1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.G2.state_dict(), f"{self.save_dir}/G2_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.D1.state_dict(), f"{self.save_dir}/D1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.D2.state_dict(), f"{self.save_dir}/D2_ep{epoch}_{self.opt.experiment_name}.pth")

import itertools
from models.unit_networks import *
from data.storage import *
import torch
from collections import OrderedDict
import os
import sys
import itertools
import torch
from util.image_pool import ImagePool
from util.util import *
from .base_model import BaseModel, OrderedDict
from . import networks
import os
import pyiqa
from .vgg import VGG
from loss_functions.attention import Self_Attn
from metrics.FID import *
from loss_functions.perceptual_loss import perceptual_similarity_loss
from loss_functions.texture_loss import texture_loss
from metrics.mse_psnr_ssim_vif import *
from models.networks import init_net
from piq import brisque
import matplotlib.patches as patches
from torchmetrics.functional.image import image_gradients

def compute_kl(mu):
    mu_2 = torch.pow(mu, 2)
    loss = torch.mean(mu_2)
    return loss


def save_fig_to_pdf(img, idx, tag, bbox):
    image_array = img.cpu().numpy()  # Assuming image_tensor is a PyTorch tensor

    # Create a figure without axis
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off axis labels and ticks

    # Plot the image
    ax.imshow(image_array[0, 0, :, :], cmap='gray')

    if bbox:
        # Create a rectangle patch
        rect = patches.Rectangle((100, 148), 50, 50, linewidth=1, edgecolor='r', facecolor='none')  # test 2: 158, test 3,4: 148

        # Add the rectangle to the plot
        ax.add_patch(rect)
    # plt.show()
    # Save the figure as a PDF without axis
    plt.savefig(f'{tag}_{idx}.pdf', format='pdf', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()  # Close the figure to free up memory


class UNITModel(BaseModel):

    def __init__(self, opt):
        """Initialize the UNIT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterionTexture = torch.nn.L1Loss(reduction='none')
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
        if opt.texture_criterion == 'attention':
            self.attention = init_net(Self_Attn(1, 'relu'))
            self.weight_X1 = list()
            self.weight_X2 = list()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.E1.parameters(), self.E2.parameters(),
                                                                self.G1.parameters(), self.G2.parameters(),
                                                                self.attention.parameters()),
                                                lr=opt.lr,
                                                betas=(opt.b1, opt.b2),
                                                )
            self.model_names = ['E1', 'E2', 'G1', 'G2', 'D1', 'D2', "attention"]
        else:
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.E1.parameters(), self.E2.parameters(), self.G1.parameters(), self.G2.parameters()),
                lr=opt.lr,
                betas=(opt.b1, opt.b2),
            )
            self.model_names = ['E1', 'E2', 'G1', 'G2', 'D1', 'D2']

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

        if opt.lambda_perceptual > 0.0:
            if opt.vgg_pretrained:
                self.vgg = VGG().eval()
            else:
                self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned', saved_weights_path=opt.vgg_model_path).eval()

        # dictionary to store training loss
        if opt.lambda_texture > 0:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2', "cycle_texture_X1", "cycle_texture_X2"]
        elif opt.lambda_perceptual > 0:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2', "perceptual_X1", "perceptual_X2"]
        else:
            self.loss_names = ['GAN_1', 'GAN_2', 'KL_1', 'KL_2', 'ID_1', 'ID_2', 'KL_1_', 'KL_2_', 'cyc_1', 'cyc_2']

        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        # dictionary to store metrics
        self.metric_names = ['psnr', 'mse', 'ssim', 'vif', 'NIQE', 'FID', 'brisque']
        self.metrics_eval = OrderedDict()
        for key in self.metric_names:
            self.metrics_eval[key] = list()

        # dictionary to store metrics per patient

        # metrics initialization
        self.fid_object = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64)
        self.real_test_buffer = []
        self.fake_test_buffer = []
        self.raps = list()
        # NIQE metric
        self.niqe = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)

        # Folders initialization
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.img_dir = os.path.join(self.web_dir, f'{opt.image_folder}')
        self.test_dir = os.path.join(self.web_dir, f'{opt.test_folder}')
        self.loss_dir = os.path.join(self.web_dir, f'{opt.loss_folder}')
        self.metric_dir = os.path.join(self.web_dir, f'{opt.metric_folder}')
        print('create web directory %s...' % self.web_dir)
        mkdirs([self.web_dir, self.img_dir]) if (opt.image_folder is not None) else "pass"
        mkdirs([self.web_dir, self.loss_dir]) if (opt.loss_folder is not None) else "pass"
        mkdirs([self.web_dir, self.test_dir]) if (opt.test_folder is not None) else "pass"
        mkdirs([self.web_dir, self.metric_dir]) if (opt.metric_folder is not None) else "pass"

        self.opt = opt

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        if self.opt.dataset_mode == "LIDC_IDRI":
            self.X1 = input['img'].type(self.Tensor).expand(-1, 3, -1, -1).to(self.device)
            self.id = input['patient'][0]
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

    def test(self):
        with torch.no_grad():
            _, self.Z1 = self.E1(self.X1)
            self.fake_X2 = self.G2(self.Z1)
            self.compute_metrics()
            self.track_metrics()
            print(self.fake_X2.shape)

    def test_visuals(self, iter, list_index):

        with torch.no_grad():
            if iter in list_index:
                _, self.Z1 = self.E1(self.X1)
                self.fake_X2 = self.G2(self.Z1)
                grad_fake_X2_x, grad_fake_X2_y = image_gradients(self.fake_X2.cpu())
                grad_fake_B = torch.sqrt(grad_fake_X2_x ** 2 + grad_fake_X2_y ** 2)
                save_fig_to_pdf(self.fake_X2, iter, self.opt.experiment_name, False)
                save_fig_to_pdf(grad_fake_B, iter, f"grad_{self.opt.experiment_name}", True)
                save_fig_to_pdf(grad_fake_B[:, :, 148:198, 100:150], iter, f"zoom_{self.opt.experiment_name}", False)  # test 2: 158:208, test 3,4: 148:198

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

        # Texture Loss
        if self.opt.lambda_texture > 0:
            if self.opt.texture_criterion == 'attention':
                loss_texture_X1, _, weight_X1 = texture_loss(self.recon_X1, self.X1, self.criterionTexture, self.opt, self.attention)
                loss_texture_X2, _, weight_X2 = texture_loss(self.recon_X2, self.X2, self.criterionTexture, self.opt, self.attention)
                print(loss_texture_X2)
                self.loss_cycle_texture_X1 = loss_texture_X1
                self.loss_cycle_texture_X2 = loss_texture_X2
                self.weight_X1.append(weight_X1.item())
                self.weight_X2.append(weight_X2.item())

            elif self.opt.texture_criterion == 'max':
                loss_texture_X1, _, _ = texture_loss(self.recon_X1, self.X1, self.criterionTexture, self.opt)
                loss_texture_X2, _, _ = texture_loss(self.recon_X2, self.X2, self.criterionTexture, self.opt)

                self.loss_cycle_texture_X1 = loss_texture_X1 * self.opt.lambda_texture
                self.loss_cycle_texture_X2 = loss_texture_X2 * self.opt.lambda_texture

            elif self.opt.texture_criterion == 'average':
                loss_texture_X1 = texture_loss(self.recon_X1, self.X1, self.criterionTexture, self.opt)
                loss_texture_X2 = texture_loss(self.recon_X2, self.X2, self.criterionTexture, self.opt)

                self.loss_cycle_texture_X1 = loss_texture_X1 * self.opt.lambda_texture
                self.loss_cycle_texture_X2 = loss_texture_X2 * self.opt.lambda_texture

            elif self.opt.texture_criterion == 'Frobenius':
                loss_texture_X1 = texture_loss(self.recon_X1, self.X1, self.criterionTexture, self.opt)
                loss_texture_X2 = texture_loss(self.recon_X2, self.X2, self.criterionTexture, self.opt)

                self.loss_cycle_texture_X1 = loss_texture_X1 * self.opt.lambda_texture
                self.loss_cycle_texture_X2 = loss_texture_X2 * self.opt.lambda_texture
            else:
                raise NotImplementedError
        else:
            self.loss_cycle_texture_X1 = 0
            self.loss_cycle_texture_X2 = 0

        # Perceptual loss
        if self.opt.lambda_perceptual > 0:
            loss_perceptual_X1 = perceptual_similarity_loss(self.recon_X1, self.X1, self.vgg, self.opt.perceptual_layers)
            loss_perceptual_X2 = perceptual_similarity_loss(self.recon_X2, self.X2, self.vgg, self.opt.perceptual_layers)
            self.loss_perceptual_X1 = loss_perceptual_X1 * self.opt.lambda_perceptual
            self.loss_perceptual_X2 = loss_perceptual_X2 * self.opt.lambda_perceptual
        else:
            self.loss_perceptual_X1 = 0
            self.loss_perceptual_X2 = 0

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
                + self.loss_cycle_texture_X1
                + self.loss_cycle_texture_X2
                + self.loss_perceptual_X1
                + self.loss_perceptual_X2

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
    def train(self):
        """Make models train mode after test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

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

    def print_time(self, epoch, delta_time):
        message = 'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, self.opt.n_epochs, delta_time)
        sys.stdout.write(message)

    def plot_current_losses(self, epoch, tracked_losses, name_title):
        loss_path = os.path.join(self.loss_dir, f'epoch.png')
        if self.opt.n_epochs == epoch:
            save_ordered_dict_as_csv(tracked_losses, f'{self.loss_dir}/loss.csv')
        save_plots(tracked_losses, loss_path, name_title)  # save the plots to the disk

    def save_networks(self, epoch):
        torch.save(self.E1.state_dict(), f"{self.save_dir}/E1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.E2.state_dict(), f"{self.save_dir}/E2_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.G1.state_dict(), f"{self.save_dir}/G1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.G2.state_dict(), f"{self.save_dir}/G2_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.D1.state_dict(), f"{self.save_dir}/D1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.D2.state_dict(), f"{self.save_dir}/D2_ep{epoch}_{self.opt.experiment_name}.pth")

    def setup1(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.load_networks_1(opt.epoch, opt.experiment_name)
        # self.print_networks(opt.verbose)

    def setup2(self, opt, custom_dir):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            self.load_networks_2(opt.epoch, opt.experiment_name, custom_dir)
        # self.print_networks(opt.verbose)

    def load_networks_2(self,epoch, exp, custom_dir):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = 'net_%s_ep%s_%s.pth' % (name, epoch, exp)
                load_path = os.path.join(custom_dir, load_filename)
                net = getattr(self, name)
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                print(state_dict.keys())
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Remove the specific word or prefix you want to eliminate
                    new_key = key.replace('module.', '')
                    new_state_dict[new_key] = value
                print(new_state_dict.keys())

                # if hasattr(state_dict, '_metadata'):
                #    del new_state_dict._metadata

                """# patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))"""
                net.load_state_dict(new_state_dict)

    def load_networks_1(self, epoch, exp):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = 'net_%s_ep%s_unit_%s.pth' % (name, epoch, exp)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                # if isinstance(net, torch.nn.DataParallel):
                #     net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                print(state_dict.keys())
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Remove the specific word or prefix you want to eliminate
                    new_key = key.replace('module.', '')
                    new_state_dict[new_key] = value
                print(new_state_dict.keys())

                # if hasattr(state_dict, '_metadata'):
                #    del new_state_dict._metadata

                """# patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))"""
                net.load_state_dict(new_state_dict)

    def compute_metrics(self):
        if self.opt.dataset_mode == "LIDC_IDRI":
            # NIQE
            fake_X2_3channels = self.fake_X2.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_X2_3channels).item()
            self.raps.append(azimuthalAverage(np.squeeze(self.fake_X2[0, 0, :, :].cpu().detach().numpy())).tolist())
            self.brisque = 0
            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0
        elif self.opt.test == "elcap_complete":
            # NIQE
            fake_X2_3channels = self.fake_X2.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_X2_3channels).item()
            self.raps.append(azimuthalAverage(np.squeeze(self.fake_X2[0, 0, :, :].cpu().detach().numpy())).tolist())
            self.brisque = 0
            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0
        else:
            x = tensor2im2(self.X2)
            y = tensor2im2(self.fake_X2)
            # MSE
            self.mse = mean_squared_error(x, y)
            # PSNR
            self.psnr = peak_signal_to_noise_ratio(x, y)
            # SSIM
            self.ssim = structural_similarity_index(x, y)
            # NIQE
            fake_X2_3channels = self.fake_X2.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_X2_3channels).item()
            self.brisque = 0
            # VIF
            self.vif = vif(x, y)
            self.fake_test_buffer.append(self.fake_X2)
            self.real_test_buffer.append(self.X2)

    def compute_FrechetInceptionDistance(self):
        real_buffer = torch.cat(self.real_test_buffer, dim=0)
        fake_buffer = torch.cat(self.fake_test_buffer, dim=0)

        fid_score = self.fid_object.compute_fid(fake_buffer, real_buffer, self.opt.dataset_len)
        self.metrics_eval['FID'].append(fid_score)

    def track_metrics(self):
        for name in self.metric_names:
            if isinstance(name, str) and name != 'FID':
                self.metrics_eval[name].append(
                    float(getattr(self, name)))  # float(...) works for both scalar tensor and float number

    def get_epoch_performance(self):
        return self.metrics_eval

    def save_metrics(self, epoch_performance, epoch):
        csv_path2 = os.path.join(self.metric_dir, f'metrics_{self.opt.test}_epoch{epoch}.csv')
        save_ordered_dict_as_csv(epoch_performance, csv_path2)
        empty_dictionary(self.metrics_eval)

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []

import torch

from models.standardGAN_networks import *
import itertools
from util.util import *
from .base_model import BaseModel, OrderedDict
import pyiqa
from metrics.FID import *
from metrics.mse_psnr_ssim_vif import *
import sys


def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


class StandardGANModel(BaseModel):
    def __init__(self, opt):
        """Initialize the StandardGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        self.c_dim = 2
        self.img_shape = (opt.channels, opt.img_height, opt.img_width)
        self.Tensor = torch.FloatTensor if str(self.device).find("cpu") != -1 else torch.Tensor

        self.criterion_L1 = torch.nn.L1Loss().to(self.device)
        self.adversarial_loss = torch.nn.MSELoss().to(self.device)

        # Content encoder, Content decoder
        self.Ec = ContentEncoder(img_shape=self.img_shape, res_blocks=opt.residual_blocks, c_dim=self.c_dim).to(self.device)
        self.Dc = ContentDecoder(channels=opt.channels, curr_dim=256).to(self.device)

        # Style encoders
        self.EsA = StyleEncoder().to(self.device)
        self.EsB = StyleEncoder().to(self.device)

        # Discriminator
        self.Disc = Discriminator(img_shape=self.img_shape, c_dim=self.c_dim).to(self.device)

        self.Ec.apply(weights_init_normal)
        self.Dc.apply(weights_init_normal)
        self.EsA.apply(weights_init_normal)
        self.EsB.apply(weights_init_normal)
        self.Disc.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.Ec.parameters(), self.Dc.parameters(), self.EsA.parameters(), self.EsB.parameters()),
                                            lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.Disc.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        # AdaIN params
        self.mean_s = 0
        self.std_s = 0

        # Dictionary to store training losses
        self.loss_names = ['cross_A', 'cross_B', 'self_A', 'self_B', 'G_cls_A', 'G_cls_B', 'G_adv_A', 'G_adv_B']
        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        # dictionary to store metrics
        self.metric_names = ['psnr', 'mse', 'ssim', 'vif', 'NIQE', 'FID', 'brisque']
        self.metrics_eval = OrderedDict()
        for key in self.metric_names:
            self.metrics_eval[key] = list()

        # metrics initialization
        self.fid_object = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64)
        self.real_test_buffer = []
        self.fake_test_buffer = []
        self.raps = list()
        # NIQE metric
        self.niqe = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)

        # Test buffers
        self.standardized_test_buffer = []
        self.not_standardized_test_buffer = []

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

    def set_input(self, sample):
        # Model inputs
        if self.opt.isTrain:
            self.imgs_sA = sample["A"].expand(-1, 3, -1, -1).type(self.Tensor).to(self.device)
            self.label_sA = sample['label_A'].type(self.Tensor).to(self.device)
            self.imgs_sB = sample['B'].expand(-1, 3, -1, -1).type(self.Tensor).to(self.device)
            self.label_sB = sample['label_B'].type(self.Tensor).to(self.device)
        else:
            self.img = sample["img"].expand(-1, 5, -1, -1).type(self.Tensor).to(self.device)

    def test(self):
        with torch.no_grad():
            self.latent_test = self.Ec(self.img, None, None, False)
            self.standardized_img = self.Dc(self.latent_test, self.mean_s, self.std_s)
            self.not_standardized_test_buffer.append(self.img)
            self.standardized_test_buffer.append(self.standardized_img)

    def forward(self):
        self.optimizer_G.zero_grad()
        # Domain A content conditioned to domain A label, domain A content conditioned to domain B label
        self.latent_A_lA, self.latent_A_lB = self.Ec(self.imgs_sA, self.label_sA, self.label_sB)
        # Domain A AdaIN params
        self.mean_sA, self.std_sA = self.EsA(self.imgs_sA)

        # Identity reconstruction Domain A
        self.identity_AA = self.Dc(self.latent_A_lA, self.mean_sA, self.std_sA)

        # Domain B content conditioned to domain B label, domain B content conditioned to domain A label
        self.latent_B_lB, self.latent_B_lA = self.Ec(self.imgs_sB, self.label_sB, self.label_sA)
        # Domain B AdaIN params
        self.mean_sB, self.std_sB = self.EsB(self.imgs_sB)

        # Identity reconstruction Domain B
        self.identity_BB = self.Dc(self.latent_B_lB, self.mean_sB, self.std_sB)

        # Overall AdaIN params
        # self.combine_adain_params([self.mean_sA, self.mean_sB], [self.std_sA, self.std_sB])

        # Fake sample in domain B using domain A latent
        self.fake_AB = self.Dc(self.latent_A_lB, self.mean_sB, self.std_sB)

        # Fake sample in domain A using domain B latent
        self.fake_BA = self.Dc(self.latent_B_lA, self.mean_sA, self.std_sA)

        # Domain B fake content conditioned to domain A label
        self.latent_fake_B_lA, _ = self.Ec(self.fake_AB, self.label_sA, self.label_sB)
        # Domain B fake AdaIN params
        self.mean_sB, self.std_sB = self.EsB(self.fake_AB)

        # Domain A fake content conditioned to domain B label
        self.latent_fake_A_lB, _ = self.Ec(self.fake_BA, self.label_sB, self.label_sA)
        # Domain A fake AdaIN params
        self.mean_sA, self.std_sA = self.EsA(self.fake_BA)

        # Overall AdaIN params
        # self.combine_adain_params([self.mean_sA_fake, self.mean_sB_fake], [self.std_sA_fake, self.std_sB_fake])

        # Reconstruction in domain A
        self.recon_A = self.Dc(self.latent_fake_B_lA, self.mean_sA, self.std_sA)
        # Reconstruction in domain B
        self.recon_B = self.Dc(self.latent_fake_A_lB, self.mean_sB, self.std_sB)

    def combine_adain_params(self, means, stds):
        self.mean_s = sum(means) / len(means)
        self.std_s = sum(stds) / len(stds)

    def backward_G(self):
        self.loss_cross_A = self.criterion_L1(self.imgs_sA, self.recon_A) * self.opt.lambda_1
        self.loss_cross_B = self.criterion_L1(self.imgs_sB, self.recon_B) * self.opt.lambda_1
        self.loss_self_A = self.criterion_L1(self.imgs_sA, self.identity_AA) * self.opt.lambda_2
        self.loss_self_B = self.criterion_L1(self.imgs_sA, self.identity_BB) * self.opt.lambda_2

        # Discriminator evaluates translated image domain A
        self.fake_validity_A, self.pred_cls_A = self.Disc(self.fake_BA)

        # Adversarial loss domain A
        self.loss_G_adv_A = self.adversarial_loss(self.fake_validity_A, torch.ones_like(self.fake_validity_A)) * self.opt.lambda_4
        # Classification loss domain A
        self.loss_G_cls_A = criterion_cls(self.pred_cls_A, self.label_sA) * self.opt.lambda_3

        # Discriminator evaluates translated image domain B
        self.fake_validity_B, self.pred_cls_B = self.Disc(self.fake_AB)
        # Adversarial loss domain B
        self.loss_G_adv_B = self.adversarial_loss(self.fake_validity_B, torch.ones_like(self.fake_validity_B)) * self.opt.lambda_4
        # Classification loss domain B
        self.loss_G_cls_B = criterion_cls(self.pred_cls_B, self.label_sB) * self.opt.lambda_3

        self.loss_G = (
                self.loss_cross_A
                + self.loss_cross_B
                + self.loss_self_A
                + self.loss_self_B
                + self.loss_G_cls_A
                + self.loss_G_cls_B
                + self.loss_G_adv_A
                + self.loss_G_adv_B
        )

        self.loss_G.backward()

    def backward_D(self):
        # Real images discrimination
        self.real_validity_A, self.pred_cls_A = self.Disc(self.imgs_sA)
        self.real_validity_B, self.pred_cls_B = self.Disc(self.imgs_sB)
        self.fake_validity_B, self.pred_cls_B = self.Disc(self.fake_AB.detach())

        # Adversarial loss
        self.loss_D_adv_A = self.adversarial_loss(self.real_validity_A, torch.ones_like(self.real_validity_A)) + self.adversarial_loss(self.fake_validity_A.detach(),
                                                                                                                                       torch.zeros_like(self.fake_validity_A))
        self.loss_D_adv_B = self.adversarial_loss(self.real_validity_B, torch.ones_like(self.real_validity_B)) + self.adversarial_loss(self.fake_validity_B.detach(),
                                                                                                                                       torch.zeros_like(self.fake_validity_B))

        # Classification loss
        self.loss_D_cls_A = criterion_cls(self.pred_cls_A, self.label_sA)
        self.loss_D_cls_B = criterion_cls(self.pred_cls_B, self.label_sB)

        # Total loss
        self.loss_D = (
                self.loss_D_adv_A * self.opt.lambda_4
                + self.loss_D_adv_B * self.opt.lambda_4
                + self.loss_D_cls_A * self.opt.lambda_3
                + self.loss_D_cls_B * self.opt.lambda_3
        )

        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.combine_adain_params([self.mean_sA, self.mean_sB], [self.std_sA, self.std_sB])
    #############################

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
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s AdaIN mean: %s, AdaIN std: %s"
            % (epoch, self.opt.n_epochs, dataset_len, i, self.loss_D.item(), self.loss_G.item(), time_left, self.mean_s, self.std_s)
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
        torch.save(self.Ec.state_dict(), f"{self.save_dir}/E1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.Dc.state_dict(), f"{self.save_dir}/E2_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.Disc.state_dict(), f"{self.save_dir}/G1_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.EsA.state_dict(), f"{self.save_dir}/G2_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.EsB.state_dict(), f"{self.save_dir}/D1_ep{epoch}_{self.opt.experiment_name}.pth")

    def save_test_images(self, epoch):
        standardized_test = torch.cat(self.standardized_test_buffer, dim=0)
        not_standardized_test = torch.cat(self.not_standardized_test_buffer, dim=0)
        torch.save(standardized_test, f'{self.test_dir}/standardized_{self.opt.test}_epoch{epoch}.pth')
        torch.save(not_standardized_test, f'{self.test_dir}/not_standardized_{self.opt.test}_epoch{epoch}.pth')


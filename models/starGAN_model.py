from util.util import *
import pyiqa
from metrics.FID import *
from metrics.mse_psnr_ssim_vif import *
import sys
from data.storage import *
from .vgg import VGG
from torch.autograd import Variable
import torch.autograd as autograd
from models.starGAN_networks import *
from loss_functions.attention import Self_Attn
from .base_model import BaseModel, OrderedDict
from loss_functions.glcm_soft_einsum_ import _texture_loss, _GridExtractor, _texture_loss_d5, _GridExtractor_d5
import torch.nn.functional as F
import torch
from models.networks import init_net
from loss_functions.perceptual_loss import perceptual_similarity_loss
import itertools

def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


def compute_gradient_penalty(T, D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = T(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).to(real_samples)

    d_interpolates, _ = D(interpolates)
    fake = Variable(T(np.ones(d_interpolates.shape)), requires_grad=False).to(real_samples)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class StarGANModel(BaseModel):
    def __init__(self, opt):
        """Initialize the StarGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        self.c_dim = 1
        self.img_shape = (opt.channels, opt.img_height, opt.img_width)
        self.Tensor = torch.FloatTensor if str(self.device).find("cpu") != -1 else torch.Tensor

        self.criterion_cycle = torch.nn.L1Loss().to(self.device)
        self.net_generator = GeneratorResNet(img_shape=self.img_shape, res_blocks=opt.residual_blocks, c_dim=self.c_dim).to(self.device)
        self.net_discriminator = Discriminator(img_shape=self.img_shape, c_dim=self.c_dim).to(self.device)

        self.net_generator.apply(weights_init_normal)
        self.net_discriminator.apply(weights_init_normal)

        # Dictionary to store training losses
        self.loss_names = ['D_adv', 'D_cls', 'G_adv', 'G_cls', 'G_rec']
        # specify the training losses you want to print out. The training/test scripts will call
        if opt.experiment_name.find('texture') != -1:
            self.loss_names = ['D_adv', 'D_cls', 'G_adv', 'G_cls', 'G_rec', 'cycle_texture']
        elif opt.experiment_name.find('perceptual') != -1:
            self.loss_names = ['D_adv', 'D_cls', 'G_adv', 'G_cls', 'G_rec', 'perceptual']
        elif opt.experiment_name.find('baseline') != -1:
            self.loss_names = ['D_adv', 'D_cls', 'G_adv', 'G_cls', 'G_rec']

        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()

        # Optimizers
        if opt.texture_criterion == 'attention':
            self.attention = init_net(Self_Attn(1, 'relu'))
            self.weight = list()
            self.attention_map = list()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.net_generator.parameters(), self.attention.parameters()),
                                                lr=opt.lr, betas=(opt.b1, opt.b2),
                                                )
            self.model_names = ['net_generator', 'net_discriminator', 'attention']
        else:
            self.optimizer_G = torch.optim.Adam(self.net_generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
            self.model_names = ['net_generator', 'net_discriminator']

        self.optimizer_D = torch.optim.Adam(self.net_discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        if opt.lambda_perceptual > 0.0:
            if opt.vgg_pretrained:
                self.vgg = VGG().eval()
            else:
                self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned', saved_weights_path=opt.vgg_model_path).eval()

        self.grid_extractor = _GridExtractor()
        # Wrap the texture_extractor in DataParallel if you have multiple GPUs
        if torch.cuda.device_count() > 1:
            self.grid_extractor = nn.DataParallel(self.grid_extractor)


        # dictionary to store metrics
        self.metric_names = ['psnr', 'mse', 'ssim', 'vif', 'NIQE', 'FID', 'brisque']
        self.metrics_eval = OrderedDict()
        for key in self.metric_names:
            self.metrics_eval[key] = list()

        # dictionary to store metrics per patient
        self.metrics_data_1 = init_storing(test_1_ids, self.metric_names)
        self.metrics_data_2 = init_storing(test_2_ids, self.metric_names)
        self.metrics_data_3 = init_storing(test_3_ids, self.metric_names)
        self.raps_data_3 = init_storing(test_3_ids, ['raps'])
        self.metrics_data_4 = init_storing(test_4_ids, self.metric_names)
        self.raps_data_4 = init_storing(test_4_ids, ['raps'])

        # metrics initialization
        self.fid_object = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64)
        self.real_test_buffer = []
        self.fake_test_buffer = []
        self.raps = list()

        # NIQE metric
        self.niqe = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)

        # Test buffers
        self.real_buffer_2 = init_image_buffer(test_2_ids)
        self.fake_buffer_2 = init_image_buffer(test_2_ids)

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

    def set_input(self, batch):
        sample = batch[0]
        self.iteration = batch[1]
        if self.opt.isTrain or (not self.opt.isTrain and (self.opt.test == "test_3" or self.opt.test == "elcap_complete")):
            self.imgs = sample["img"].type(self.Tensor).expand(-1, 3, -1, -1).to(self.device)
            self.label = sample["label"].type(self.Tensor).to(self.device)
            self.id = sample["patient"][0]
            # Sample labels as generator inputs
            self.sampled_c = self.Tensor(np.random.randint(0, 2, (self.imgs.size(0), self.c_dim))).to(self.device)
        else:
            self.img_A = sample["img_A"].type(self.Tensor).expand(-1, 3, -1, -1).to(self.device)
            self.img_B = sample["img_B"].type(self.Tensor).expand(-1, 3, -1, -1).to(self.device)
            self.label_A = sample["label_A"].type(self.Tensor).to(self.device)
            self.label_B = sample["label_B"].type(self.Tensor).to(self.device)
            self.id = sample["patient"][0]

    def test(self):
        with torch.no_grad():
            if self.opt.test == "test_1" or self.opt.test == "test_2":
                self.fake_B = self.net_generator(self.img_A, self.label_B)
            elif self.opt.test == "test_3" or self.opt.test == "elcap_complete":
                self.fake_B = self.net_generator(self.imgs, self.label)
            self.compute_metrics()
            self.track_metrics_per_patient(self.id)

    def forward(self):
        # Generate fake batch of images
        self.fake_imgs = self.net_generator(self.imgs, self.sampled_c)

        self.optimizer_D.zero_grad()
        # Real images
        self.real_validity, self.pred_cls = self.net_discriminator(self.imgs)
        # Fake images
        self.fake_validity, _ = self.net_discriminator(self.fake_imgs.detach())

    def backward_D(self):
        # Gradient penalty
        gradient_penality = compute_gradient_penalty(self.Tensor, self.net_discriminator, self.imgs, self.fake_imgs)
        # Adversarial loss
        self.loss_D_adv = -torch.mean(self.real_validity) + torch.mean(self.fake_validity) + self.opt.lambda_gb * gradient_penality
        # Classification loss
        self.loss_D_cls = criterion_cls(self.pred_cls, self.label)
        # Total loss
        self.loss_D = (
                self.loss_D_adv
                + self.opt.lambda_cls * self.loss_D_cls
        )

        self.loss_D.backward()

    def backward_G(self):
        # Translate and reconstruct image
        self.gen_imgs = self.net_generator(self.imgs, self.sampled_c)
        self.recov_imgs = self.net_generator(self.gen_imgs, self.label)
        # Discriminator evaluates translated image
        self.fake_validity, self.pred_cls = self.net_discriminator(self.gen_imgs)
        # Adversarial loss
        self.loss_G_adv = -torch.mean(self.fake_validity)
        # Classification loss
        self.loss_G_cls = criterion_cls(self.pred_cls, self.sampled_c)
        # Reconstruction loss
        self.loss_G_rec = self.criterion_cycle(self.recov_imgs, self.imgs)
        lambda_texture = self.opt.lambda_texture

        # Texture loss
        if lambda_texture > 0:
            if self.opt.texture_criterion == 'attention':
                loss_texture, attention_map, weight = _texture_loss(self.recov_imgs[:, 0, :, :].unsqueeze(1), self.imgs[:, 0, :, :].unsqueeze(1), self.opt, self.grid_extractor, self.attention)
                self.loss_cycle_texture = loss_texture
                self.weight.append(weight.item())
                self.attention_map.append(attention_map.detach().clone().cpu().numpy())


            elif self.opt.texture_criterion == 'max':
                loss_texture = _texture_loss(self.recov_imgs[:, 0, :, :].unsqueeze(1), self.imgs[:, 0, :, :].unsqueeze(1), self.opt, self.grid_extractor)
                plt.imshow(self.recov_imgs[0, 0, :, :].detach().numpy())
                plt.show()
                plt.imshow(self.imgs[0, 0, :, :].detach().numpy())
                plt.show()
                # compute the loss function by averaging over the batch
                self.loss_cycle_texture = loss_texture * lambda_texture

            elif self.opt.texture_criterion == 'average':
                loss_texture = _texture_loss(self.recov_imgs[:, 0, :, :].unsqueeze(1), self.imgs[:, 0, :, :].unsqueeze(1), self.opt, self.grid_extractor)

                self.loss_cycle_texture = loss_texture * lambda_texture

            elif self.opt.texture_criterion == 'Frobenius':
                loss_texture = _texture_loss(self.recov_imgs[:, 0, :, :].unsqueeze(1), self.imgs[:, 0, :, :].unsqueeze(1), self.opt, self.grid_extractor)

                self.loss_cycle_texture = loss_texture * lambda_texture
            else:
                raise NotImplementedError
        else:
            self.loss_cycle_texture = 0

        # Perceptual loss
        lambda_perceptual = self.opt.lambda_perceptual
        if lambda_perceptual > 0:
            loss_perceptual = perceptual_similarity_loss(self.recov_imgs, self.imgs, self.vgg,
                                                           self.opt.perceptual_layers)

            self.loss_perceptual = loss_perceptual * lambda_perceptual

        else:
            self.loss_perceptual = 0


        # Total loss
        self.loss_G = (
                self.loss_G_adv
                + self.opt.lambda_cls * self.loss_G_cls
                + self.opt.lambda_rec * self.loss_G_rec
                + self.loss_cycle_texture
                + self.loss_perceptual
        )

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        if self.iteration % self.opt.n_critic == 0:
            self.backward_G()
            self.optimizer_G.step()

    ######################################################
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
            % (epoch, self.opt.n_epochs, dataset_len, i, self.loss_D.item(), self.loss_G.item(), time_left)
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
        torch.save(self.net_generator.state_dict(), f"{self.save_dir}/net_gen_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.net_discriminator.state_dict(), f"{self.save_dir}/net_disc_ep{epoch}_{self.opt.experiment_name}.pth")

    def load_networks_2(self, epoch, exp, custom_dir):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                # load_filename = '%s_net_%s_%s.pth' % (epoch, name, exp)
                load_filename = "net_gen_ep_50_baseline_1.pth"
                load_path = os.path.join(custom_dir, load_filename)
                net = getattr(self, 'net_generator')
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def compute_metrics(self):
        if self.opt.test == "test_3":
            # NIQE
            fake_B_3channels = self.fake_B.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_B_3channels).item()
            self.raps = azimuthalAverage(np.squeeze(self.fake_B[0, 0, :, :].cpu().detach().numpy())).tolist()
            self.brisque = 0
            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0

        elif self.opt.test == "elcap_complete":
            # NIQE
            fake_B_3channels = self.fake_B.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_B_3channels).item()
            self.raps = azimuthalAverage(np.squeeze(self.fake_B[0, 0, :, :].cpu().detach().numpy())).tolist()
            self.brisque = 0
            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0

        elif self.opt.test == "test_2":
            x = tensor2im2(self.img_B)
            y = tensor2im2(self.fake_B)
            # MSE
            self.mse = mean_squared_error(x, y)
            # PSNR
            self.psnr = peak_signal_to_noise_ratio(x, y)
            # SSIM
            self.ssim = structural_similarity_index(x, y)
            # NIQE
            fake_B_3channels = self.fake_B.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_B_3channels).item()

            self.brisque = 0
            # VIF
            self.vif = vif(x, y)

            self.fake_buffer_2[self.id].append(self.fake_B)
            print(self.fake_buffer_2)
            self.real_buffer_2[self.id].append(self.img_B)

        elif self.opt.test == "test_1":
            x = tensor2im2(self.img_B)
            y = tensor2im2(self.fake_B)
            # MSE
            self.mse = mean_squared_error(x, y)
            # PSNR
            self.psnr = peak_signal_to_noise_ratio(x, y)
            # SSIM
            self.ssim = structural_similarity_index(x, y)
            # NIQE
            fake_B_3channels = self.fake_B.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_B_3channels).item()

            self.brisque = 0
            # VIF
            self.vif = vif(x, y)

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

    def save_raps_per_patient(self, epoch):
        if self.opt.test == "test_3":
            save_to_json(self.raps_data_3, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.raps_data_3, nesting=2)
        elif self.opt.test == "elcap_complete":
            save_to_json(self.raps_data_4, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.raps_data_4, nesting=2)

    def fid_compute(self):

        for key in self.real_buffer_2.keys():
            real_buffer = torch.cat(self.real_buffer_2[key], dim=0)
            fake_buffer = torch.cat(self.fake_buffer_2[key], dim=0)

            fid_score = self.fid_object.compute_fid(fake_buffer, real_buffer, self.opt.dataset_len)
            self.metrics_data_2[key]['FID'].append(fid_score)

        # torch.save(real_buffer, f'{self.test_dir}/real_buffer_{self.opt.test}_epoch{epoch}.pth')
        # torch.save(fake_buffer, f'{self.test_dir}/fake_buffer_{self.opt.test}_epoch{epoch}.pth')

        empty_Dictionary(self.real_buffer_2, nesting=1)
        empty_Dictionary(self.fake_buffer_2, nesting=1)

    def save_metrics_per_patient(self, epoch):
        if self.opt.test == "test_1":
            save_to_json(self.metrics_data_1, f"{self.metric_dir}/metrics_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.metrics_data_1, nesting=2)
        elif self.opt.test == "test_2":
            save_to_json(self.metrics_data_2, f"{self.metric_dir}/metrics_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.metrics_data_2, nesting=2)
        elif self.opt.test == "test_3":
            save_to_json(self.metrics_data_3, f"{self.metric_dir}/metrics_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.metrics_data_3, nesting=2)
        elif self.opt.test == "elcap_complete":
            save_to_json(self.metrics_data_4, f"{self.metric_dir}/metrics_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.metrics_data_4, nesting=2)

    def save_test_images(self, epoch):
        standardized_test = torch.cat(self.standardized_test_buffer, dim=0)
        not_standardized_test = torch.cat(self.not_standardized_test_buffer, dim=0)
        torch.save(standardized_test, f'{self.test_dir}/standardized_{self.opt.test}_epoch{epoch}.pth')
        torch.save(not_standardized_test, f'{self.test_dir}/not_standardized_{self.opt.test}_epoch{epoch}.pth')
        self.standardized_test_buffer = []
        self.not_standardized_test_buffer = []

    def save_metrics(self, epoch_performance, epoch):
        csv_path2 = os.path.join(self.metric_dir, f'metrics_{self.opt.test}_epoch{epoch}.csv')
        save_ordered_dict_as_csv(epoch_performance, csv_path2)
        empty_dictionary(self.metrics_eval)

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []
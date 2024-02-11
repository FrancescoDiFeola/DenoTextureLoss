from util.util import *
import pyiqa
from metrics.FID import *
from metrics.mse_psnr_ssim_vif import *
import sys
from data.storage import *
from .vgg import VGG
from torch.autograd import Variable
import torch.autograd as autograd
from models.srGAN_networks import *
from loss_functions.attention import Self_Attn
from .base_model import BaseModel, OrderedDict
from loss_functions.glcm_soft_einsum_ import _texture_loss, _GridExtractor, _texture_loss_d5, _GridExtractor_d5
import torch.nn.functional as F
import torch
from models.networks import init_net
from loss_functions.perceptual_loss import perceptual_similarity_loss
import itertools


class SRGANModel(BaseModel):
    def __init__(self, opt):
        """Initialize the StarGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        hr_shape = (opt.hr_height, opt.hr_width)
        self.FloatTensor = torch.FloatTensor if str(self.device).find("cpu") != -1 else torch.Tensor
        self.LongTensor = torch.cuda.LongTensor if str(self.device).find("cpu") != -1 else torch.LongTensor

        # Initialize generator and discriminator
        self.net_generator = GeneratorResNet().to(self.device)
        self.net_discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(self.device)

        self.feature_extractor = FeatureExtractor()
        # Set feature extractor to inference mode
        self.feature_extractor.eval()

        self.net_generator.apply(weights_init_normal)
        self.net_discriminator.apply(weights_init_normal)

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_content = torch.nn.L1Loss()

        # Dictionary to store training losses
        # specify the training losses you want to print out. The training/test scripts will call
        if opt.experiment_name.find('texture') != -1:
            self.loss_names = ['D_adv', 'G_adv', 'G_cont', 'cycle_texture']
        elif opt.experiment_name.find('perceptual') != -1:
            self.loss_names = ['D_adv', 'G_adv', 'G_cont', 'perceptual']
        elif opt.experiment_name.find('baseline') != -1:
            self.loss_names = ['D_adv', 'G_adv', 'G_cont', 'G_rec']

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

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        if self.opt.dataset_mode == "LIDC_IDRI":
            self.real_A = input['img'].type(self.FloatTensor).to(self.device)
            self.id = input['patient'][0]

        else:
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].type(self.FloatTensor).to(self.device)
            self.real_B = input['B' if AtoB else 'A'].type(self.FloatTensor).to(self.device)
            self.id = input['patient'][0]

            # Adversarial ground truths
            self.valid = Variable(self.FloatTensor(np.ones((self.real_A.size(0), *self.net_discriminator.output_shape))), requires_grad=False)
            self.fake = Variable(self.FloatTensor(np.zeros((self.real_A.size(0), *self.net_discriminator.output_shape))), requires_grad=False)

    def test(self):

        with torch.no_grad():
            self.fake_B = self.net_generator(self.real_A)
            # diff = abs(self.fake_B[0, 0, :, :] - self.real_A[0, 0, :, :])
            # self.skweness.append(skew(diff.detach().cpu().flatten()))

            self.compute_metrics()

            self.track_metrics_per_patient(self.id)

    def forward(self):
        self.optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        self.fake_B = self.net_generator(self.real_A)

    def backward_G(self):
        # Adversarial loss
        self.loss_GAN = self.criterion_GAN(self.net_discriminator(self.fake_B), self.valid)

        # MSE content loss
        self.loss_mse_content = self.criterion_content(self.fake_B, self.real_B)

        lambda_texture = self.opt.lambda_texture

        # Texture loss
        if lambda_texture > 0:
            if self.opt.texture_criterion == 'attention':
                loss_texture, attention_map, weight = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor,
                                                                    self.attention)
                self.loss_texture = loss_texture
                self.weight.append(weight.item())
                self.attention_map.append(attention_map.detach().clone().cpu().numpy())

            elif self.opt.texture_criterion == 'max':
                loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor)

                # compute the loss function by averaging over the batch
                self.loss_texture = loss_texture * lambda_texture

            elif self.opt.texture_criterion == 'average':
                loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor)

                self.loss_texture = loss_texture * lambda_texture

            elif self.opt.texture_criterion == 'Frobenius':
                loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor)

                self.loss_texture = loss_texture * lambda_texture
            else:
                raise NotImplementedError
        else:
            self.loss_texture = 0

        # Perceptual loss
        lambda_perceptual = self.opt.lambda_perceptual
        if lambda_perceptual > 0:
            loss_perceptual = perceptual_similarity_loss(self.fake_B, self.real_B, self.vgg, self.opt.perceptual_layers)

            self.loss_perceptual = loss_perceptual * lambda_perceptual

        else:
            self.loss_perceptual = 0

        # Total loss
        self.loss_G = (
                self.loss_mse_content
                + self.opt.lambda_GAN * self.loss_GAN
                + self.loss_texture
                + self.loss_perceptual
        )

        self.loss_G.backward()

    def backward_D(self):
        self.optimizer_D.zero_grad()
        # Loss of real and fake images
        loss_real = self.criterion_GAN(self.net_discriminator(self.real_B), self.valid)
        loss_fake = self.criterion_GAN(self.net_discriminator(self.real_B.detach()), self.fake)
        # Total loss
        self.loss_D = (loss_real + loss_fake) / 2
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        self.backward_G()
        self.optimizer_G.step()
        self.backward_D()
        self.optimizer_D.step()

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
            x = tensor2im2(self.real_B)
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
            self.real_buffer_2[self.id].append(self.real_B)

        elif self.opt.test == "test_1":
            x = tensor2im2(self.real_B)
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

    def save_metrics(self, epoch_performance, epoch):
        csv_path2 = os.path.join(self.metric_dir, f'metrics_{self.opt.test}_epoch{epoch}.csv')
        save_ordered_dict_as_csv(epoch_performance, csv_path2)
        empty_dictionary(self.metrics_eval)

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []

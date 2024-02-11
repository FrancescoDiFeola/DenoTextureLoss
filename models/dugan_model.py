from models.dugan_networks import *
import copy
import sys
import itertools
from data.storage import *
from util.util import *
from .base_model import BaseModel, OrderedDict
import pyiqa
from .vgg import VGG
from loss_functions.attention import Self_Attn
from metrics.FID import *
from loss_functions.glcm_soft_einsum_ import _texture_loss, _GridExtractor
from loss_functions.perceptual_loss import perceptual_similarity_loss
from metrics.mse_psnr_ssim_vif import *
from models.networks import init_net


def ls_gan(inputs, targets):
    return torch.mean((inputs - targets) ** 2)


def standard_gan(inputs, targets):
    if isinstance(targets, float):
        targets = torch.ones_like(inputs) * targets
    return F.binary_cross_entropy(inputs, targets)


def warmup(warmup_iter, cutmix_prob, n_iter):
    return min(n_iter * cutmix_prob / warmup_iter, cutmix_prob)


def turn_on_spectral_norm(module):
    module_output = module
    if isinstance(module, torch.nn.Conv2d):
        if module.out_channels != 1 and module.in_channels > 4:
            module_output = nn.utils.spectral_norm(module)
    # if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #     module_output = nn.utils.spectral_norm(module)
    for name, child in module.named_children():
        module_output.add_module(name, turn_on_spectral_norm(child))
    del module
    return module_output


def cutmix(mask_size):
    mask = torch.ones(mask_size)
    lam = np.random.beta(1., 1.)
    _, _, height, width = mask_size
    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))
    mask[:, :, y0:y1, x0:x1] = 0
    return mask


def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target


class SobelOperator(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.register_buffer('conv_x', torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, None, :, :] / 4)
        self.register_buffer('conv_y', torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None, :, :] / 4)

    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)

        grad_x = F.conv2d(x, self.conv_x, bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x, self.conv_y, bias=None, stride=1, padding=1)

        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        x = x.view(b, c, h, w)

        return x


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.sobel = SobelOperator(1e-4)

    def forward(self, pr, gt):
        gt_sobel = self.sobel(gt)
        pr_sobel = self.sobel(pr)
        grad_loss = F.l1_loss(gt_sobel, pr_sobel)
        return grad_loss


class DUGANModel(BaseModel):

    def __init__(self, opt):
        """Initialize the UNIT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)

        opt = self.opt
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.Tensor = torch.FloatTensor if str(self.device).find("cpu") != -1 else torch.Tensor

        self.generator = Generator(in_channels=1, out_channels=opt.num_channels, num_layers=opt.num_layers, kernel_size=3, padding=1).to(self.device)
        img_discriminator = UNet(repeat_num=opt.repeat_num, use_discriminator=True, conv_dim=64, use_sigmoid=False).to(self.device)
        self.img_discriminator = turn_on_spectral_norm(img_discriminator)
        # self.generator = nn.DataParallel(generator, device_ids=[0, 1])
        # self.img_discriminator = nn.DataParallel(img_discriminator, device_ids=[0, 1])

        self.sobel = SobelOperator().to(self.device)

        self.grad_discriminator = copy.deepcopy(img_discriminator)
        self.ema_generator = copy.deepcopy(self.generator)

        self.apply_cutmix_prob = torch.rand(opt.max_iter)

        self.gan_metric = ls_gan

        self.grid_extractor = _GridExtractor()
        # Wrap the texture_extractor in DataParallel if you have multiple GPUs
        if torch.cuda.device_count() > 1:
            self.grid_extractor = nn.DataParallel(self.grid_extractor)

        if opt.texture_criterion == 'attention':
            self.attention = init_net(Self_Attn(1, 'relu'))
            self.weight = list()
            self.attention_maps = list()
            self.g_optimizer = torch.optim.Adam(itertools.chain(self.generator.parameters(), self.attention.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
            self.model_names = ['generator', "attention", "img_discriminator"]
        else:
            self.g_optimizer = torch.optim.Adam(self.generator.parameters(), opt.g_lr, weight_decay=opt.weight_decay)
            self.model_names = ['generator', "img_discriminator"]

        self.img_d_optimizer = torch.optim.Adam(self.img_discriminator.parameters(), opt.d_lr)
        self.grad_d_optimizer = torch.optim.Adam(self.grad_discriminator.parameters(), opt.d_lr)

        ##########

        if opt.lambda_perceptual > 0.0:
            if opt.vgg_pretrained:
                self.vgg = VGG().eval().to("cuda:1")
            else:
                self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned', saved_weights_path=opt.vgg_model_path).eval().to("cuda:1")

        # dictionary to store training loss
        if opt.lambda_texture > 0:
            self.loss_names = ['img_D', 'grad_D', 'G', "texture"]

        elif opt.lambda_perceptual > 0:
            self.loss_names = ['img_D', 'grad_D', 'G', "perceptual"]
        else:
            self.loss_names = ['img_D', 'grad_D', 'G']

        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        # dictionary to store metrics
        self.metric_names = ['psnr', 'mse', 'ssim', 'vif', 'NIQE', 'FID_ImNet', 'FID_random', 'brisque']
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

        self.skweness_2 = init_storing(test_2_ids, ["skew"])
        self.skweness_3 = init_storing(test_3_ids, ["skew"])
        self.skweness_4 = init_storing(test_4_ids, ["skew"])

        self.difference_heatmaps_2 = init_image_buffer(test_2_ids)
        self.difference_heatmaps_3 = init_image_buffer(test_3_ids)
        self.difference_heatmaps_4 = init_image_buffer(test_4_ids)

        self.grad_real_A_2 = init_image_buffer(test_2_ids)
        self.grad_real_A_3 = init_image_buffer(test_3_ids)
        self.grad_real_A_4 = init_image_buffer(test_4_ids)
        self.grad_fake_B_2 = init_image_buffer(test_2_ids)
        self.grad_fake_B_3 = init_image_buffer(test_3_ids)
        self.grad_fake_B_4 = init_image_buffer(test_4_ids)

        self.gradient_difference_2 = init_image_buffer(test_2_ids)
        self.gradient_difference_3 = init_image_buffer(test_3_ids)
        self.gradient_difference_4 = init_image_buffer(test_4_ids)

        self.grad_correlation_2 = init_storing(test_2_ids, ["corr"])
        self.grad_correlation_3 = init_storing(test_3_ids, ["corr"])
        self.grad_correlation_4 = init_storing(test_4_ids, ["corr"])

        # metrics initialization
        self.fid_object_1 = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64, pretrained=True)
        self.fid_object_2 = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64, pretrained=False)

        self.real_buffer_2 = []  # init_image_buffer(test_2_ids)
        self.fake_buffer_2 = []  # init_image_buffer(test_2_ids)

        self.raps = 0
        self.skweness = []

        self.grid_extractor = _GridExtractor()
        # Wrap the texture_extractor in DataParallel if you have multiple GPUs
        if torch.cuda.device_count() > 1:
            self.grid_extractor = nn.DataParallel(self.grid_extractor, device_ids=[0, 1, 2, 3])

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

        # standardized buffer, not standardized buffer
        self.standardized_test_buffer = []
        self.not_standardized_test_buffer = []

        self.opt = opt

    def set_input(self, input, iter):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        if self.opt.dataset_mode == "LIDC_IDRI":
            self.real_A = input['img'].type(self.Tensor).to(self.device)
            self.image_paths = input['im_paths']
            self.id = input['patient'][0]
        else:
            # Set model input
            self.real_A = input["A"].type(self.Tensor).to(self.device)
            self.real_B = input["B"].type(self.Tensor).to(self.device)
            self.image_paths = input['A_paths']
            self.id = input['patient'][0]
        self.n_iter = iter

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.fake_B = self.generator(self.real_A)
        self.grad_fake_B = self.sobel(self.fake_B)
        self.grad_real_A = self.sobel(self.real_A)
        self.grad_real_B = self.sobel(self.real_B)

    def test(self):
        with torch.no_grad():
            self.fake_B = self.generator(self.real_A)
            self.compute_metrics()
            self.track_metrics_per_patient(self.id)

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

    def backward_img_discriminator(self):
        self.real_enc, self.real_dec = self.img_discriminator(self.real_B)
        self.fake_enc, self.fake_dec = self.img_discriminator(self.fake_B.detach())
        self.source_enc, self.source_dec = self.img_discriminator(self.real_A)

        disc_loss = self.gan_metric(self.real_enc, 1.) + self.gan_metric(self.real_dec, 1.)+self.gan_metric(self.fake_enc, 0.)+self.gan_metric(self.fake_dec, 0.)+self.gan_metric(self.source_enc, 0.)+self.gan_metric(self.source_dec, 0.)

        self.loss_img_D = disc_loss

        apply_cutmix = self.apply_cutmix_prob[self.n_iter - 1] < warmup(self.opt.cutmix_warmup_iter, self.opt.cutmix_prob, self.n_iter)
        if apply_cutmix:
            mask = cutmix(self.real_dec.size()).to(self.real_dec)

            # if random.random() > 0.5:
            #     mask = 1 - mask

            cutmix_enc, cutmix_dec = self.img_discriminator(mask_src_tgt(self.real_B, self.fake_B.detach(), mask))

            self.loss_cutmix_disc = self.gan_metric(cutmix_enc, 0.) + self.gan_metric(cutmix_dec, mask)

            loss_cr = F.mse_loss(cutmix_dec, mask_src_tgt(self.real_dec, self.fake_dec, mask))
            self.loss_cr = loss_cr
            self.loss_img_D += self.loss_cutmix_disc + self.loss_cr * self.opt.cr_loss_weight

        self.loss_img_D.backward()

    def backward_grad_discriminator(self):
        self.real_enc, self.real_dec = self.grad_discriminator(self.real_B)
        self.fake_enc, self.fake_dec = self.grad_discriminator(self.fake_B.detach())
        self.source_enc, self.source_dec = self.grad_discriminator(self.real_A)

        disc_loss = self.gan_metric(self.real_enc, 1.)+self.gan_metric(self.real_dec, 1.)+self.gan_metric(self.fake_enc, 0.) + self.gan_metric(self.fake_dec, 0.)+self.gan_metric(self.source_enc, 0.)+self.gan_metric(self.source_dec, 0.)

        self.loss_grad_D = disc_loss

        apply_cutmix = self.apply_cutmix_prob[self.n_iter - 1] < warmup(self.opt.cutmix_warmup_iter, self.opt.cutmix_prob, self.n_iter)
        if apply_cutmix:
            mask = cutmix(self.real_dec.size()).to(self.real_dec)

            # if random.random() > 0.5:
            #     mask = 1 - mask

            cutmix_enc, cutmix_dec = self.grad_discriminator(mask_src_tgt(self.real_B, self.fake_B.detach(), mask))

            self.loss_cutmix_disc = self.gan_metric(cutmix_enc, 0.) + self.gan_metric(cutmix_dec, mask)

            loss_cr = F.mse_loss(cutmix_dec, mask_src_tgt(self.real_dec, self.fake_dec, mask))
            self.loss_cr = loss_cr
            self.loss_grad_D += self.loss_cutmix_disc + self.loss_cr * self.opt.cr_loss_weight

        self.loss_grad_D.backward()

    def backward_generator(self):
        img_gen_enc, img_gen_dec = self.img_discriminator(self.fake_B)
        self.loss_img_gen = self.gan_metric(img_gen_enc, 1.) + self.gan_metric(img_gen_dec, 1.)
        self.loss_G = 0.
        if self.opt.use_grad_discriminator:
            self.grad_d_optimizer.zero_grad()
            self.backward_grad_discriminator()
            self.grad_d_optimizer.step()

            self.grad_gen_enc, self.grad_gen_dec = self.grad_discriminator(self.grad_fake_B)
            self.loss_grad_gen = self.gan_metric(self.grad_gen_enc, 1.) + self.gan_metric(self.grad_gen_dec, 1.)
            self.loss_G = self.loss_grad_gen * self.opt.grad_gen_loss_weight

        # Texture loss
        lambda_texture = self.opt.lambda_texture
        if lambda_texture > 0:
            if self.opt.texture_criterion == 'attention':
                loss_texture, att_map, weight = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor, self.attention)
                self.loss_texture = loss_texture
                self.weight.append(weight.item())
                self.attention_maps.append(att_map.detach().clone().cpu().numpy())
            elif self.opt.texture_criterion == 'max':
                loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor)  # , self.texture_extractor
                self.loss_texture = loss_texture * lambda_texture
            elif self.opt.texture_criterion == 'average':
                loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor)
                self.loss_texture = loss_texture * lambda_texture
            elif self.opt.texture_criterion == 'Frobenius':
                loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor)
                self.loss_texture = loss_texture * lambda_texture
        else:
            self.loss_texture = 0

        # Perceptual loss
        lambda_perceptual = self.opt.lambda_perceptual
        if lambda_perceptual > 0:
            loss_perceptual = perceptual_similarity_loss(self.fake_B, self.real_B, self.vgg, self.opt.perceptual_layers)
            self.loss_perceptual = loss_perceptual * lambda_perceptual
        else:
            self.loss_perceptual = 0

        ########### Pixel Loss #########
        self.loss_pix = F.mse_loss(self.fake_B, self.real_B)

        ########### L1 Loss ############
        self.loss_l1 = F.l1_loss(self.fake_B, self.real_B)

        ########### Grad Loss ############
        self.loss_grad = F.l1_loss(self.grad_fake_B, self.grad_real_B)

        self.loss_G += self.loss_img_gen * self.opt.img_gen_loss_weight+self.loss_pix * self.opt.pix_loss_weight+self.loss_grad * self.opt.grad_loss_weight+self.loss_texture+self.loss_perceptual
        self.loss_G.backward()

    def optimize_parameters(self):

        self.forward()
        self.img_d_optimizer.zero_grad()
        self.backward_img_discriminator()
        self.img_d_optimizer.step()
        if self.n_iter % self.opt.d_iter == 0:
            self.g_optimizer.zero_grad()
            self.backward_generator()
            self.g_optimizer.step()

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
            "\r[Epoch %d/%d] [Batch %d/%d] [Img_D loss: %f] [Grad_D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, self.opt.n_epochs, dataset_len, i, self.loss_img_D, self.loss_grad_D, self.loss_G, time_left)
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
        torch.save(self.generator.state_dict(), f"{self.save_dir}/net_generator_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.img_discriminator.state_dict(), f"{self.save_dir}/net_img_disc_ep{epoch}_{self.opt.experiment_name}.pth")
        torch.save(self.grad_discriminator.state_dict(), f"{self.save_dir}/net_grad_disc_ep{epoch}_{self.opt.experiment_name}.pth")
        if self.opt.texture_criterion == "attention":
            torch.save(self.attention.state_dict(), f"{self.save_dir}/net_attention_ep{epoch}_{self.opt.experiment_name}.pth")

    def setup1(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.load_networks_1(opt.epoch, opt.experiment_name)
        # self.print_networks(opt.verbose)

    def load_networks_1(self, epoch, exp):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = 'net_%s_ep%s_%s.pth' % (name, epoch, exp)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Remove the specific word or prefix you want to eliminate
                    new_key = key.replace('module.', '')
                    new_state_dict[new_key] = value

                # if hasattr(state_dict, '_metadata'):
                #    del new_state_dict._metadata

                """# patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))"""
                net.load_state_dict(new_state_dict)

    def track_metrics(self):
        for name in self.metric_names:
            if isinstance(name, str) and name != 'FID_ImNet' and name != 'FID_random':
                self.metrics_eval[name].append(
                    float(getattr(self, name)))  # float(...) works for both scalar tensor and float number

    def track_metrics_per_patient(self, id):
        if self.opt.test == "test_1":
            for name in self.metric_names:
                if isinstance(name, str) and name != 'FID_ImNet' and name != 'FID_random':
                    self.metrics_data_1[id][name].append(
                        float(getattr(self, name)))  # float(...) works for both scalar tensor and float number
            return self.metrics_data_1
        elif self.opt.test == "test_2":
            for name in self.metric_names:
                if isinstance(name, str) and name != 'FID_ImNet' and name != 'FID_random':
                    self.metrics_data_2[id][name].append(
                        float(getattr(self, name)))  # float(...) works for both scalar tensor and float number
            return self.metrics_data_2
        elif self.opt.test == "test_3":
            for name in self.metric_names:
                if isinstance(name, str) and name != 'FID_ImNet' and name != 'FID_random':
                    self.metrics_data_3[id][name].append(
                        float(getattr(self, name)))  # float(...) works for both scalar tensor and float number
            self.raps_data_3[id]['raps'].append(list(getattr(self, 'raps')))
            return self.metrics_data_3, self.raps_data_3

        elif self.opt.test == "elcap_complete":
            for name in self.metric_names:
                if isinstance(name, str) and name != 'FID':
                    self.metrics_data_4[id][name].append(
                        float(getattr(self, name)))  # float(...) works for both scalar tensor and float number

            self.raps_data_4[id]['raps'].append(list(getattr(self, 'raps')))

            return self.metrics_data_4, self.raps_data_4

    def get_epoch_performance(self):
        return self.metrics_eval

    def save_metrics(self, epoch_performance, epoch):
        csv_path2 = os.path.join(self.metric_dir, f'metrics_{self.opt.test}_epoch{epoch}.csv')
        save_ordered_dict_as_csv(epoch_performance, csv_path2)
        empty_dictionary(self.metrics_eval)

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []

    def save_attention_maps(self):
        np.save(f"{self.loss_dir}/attention.npy", np.array(self.attention_maps))

    def save_attention_weights(self):
        np.save(f"{self.loss_dir}/weight.npy", np.array(self.weight))

    def fid_compute(self):

        # for key in tqdm(self.real_buffer_2.keys()):
        # print(key)
        real_buffer = torch.cat(self.real_buffer_2, dim=0)
        fake_buffer = torch.cat(self.fake_buffer_2, dim=0)
        print(real_buffer.shape)
        print(fake_buffer.shape)
        fid_score_1 = self.fid_object_1.compute_fid(fake_buffer, real_buffer, len(real_buffer))
        fid_score_2 = self.fid_object_2.compute_fid(fake_buffer, real_buffer, len(real_buffer))
        for key in self.metrics_data_2.keys():
            self.metrics_data_2[key]['FID_ImNet'].append(fid_score_1)
            self.metrics_data_2[key]['FID_random'].append(fid_score_2)

        self.real_buffer_2 = []
        self.real_buffer_1 = []
        # empty_Dictionary(self.real_buffer_2, nesting=0)
        # empty_Dictionary(self.fake_buffer_2, nesting=0)

    def save_test_4(self, epoch):
        if self.opt.test == "test_2":
            save_to_json(self.skweness_2, f"{self.metric_dir}/skewness_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.skweness_2, nesting=2)
            save_to_json(self.grad_correlation_2, f"{self.metric_dir}/grad_correlation_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.grad_correlation_2, nesting=2)
            torch.save(self.difference_heatmaps_2, f'{self.metric_dir}/difference_heatmaps_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.difference_heatmaps_2, nesting=1)
            torch.save(self.grad_real_A_2, f'{self.metric_dir}/grad_real_A_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_real_A_2, nesting=1)
            torch.save(self.grad_fake_B_2, f'{self.metric_dir}/grad_fake_B_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_fake_B_2, nesting=1)
            torch.save(self.gradient_difference_2, f'{self.metric_dir}/gradient_difference_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.gradient_difference_2, nesting=1)

        elif self.opt.test == "test_3":
            save_to_json(self.skweness_3, f"{self.metric_dir}/skewness_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.skweness_3, nesting=2)
            save_to_json(self.grad_correlation_3, f"{self.metric_dir}/grad_correlation_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.grad_correlation_3, nesting=2)
            torch.save(self.difference_heatmaps_3, f'{self.metric_dir}/difference_heatmaps_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.difference_heatmaps_3, nesting=1)
            torch.save(self.grad_real_A_3, f'{self.metric_dir}/grad_real_A_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_real_A_3, nesting=1)
            torch.save(self.grad_fake_B_3, f'{self.metric_dir}/grad_fake_B_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_fake_B_3, nesting=1)
            torch.save(self.gradient_difference_3, f'{self.metric_dir}/gradient_difference_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.gradient_difference_3, nesting=1)
        elif self.opt.test == "elcap_complete":
            save_to_json(self.skweness_4, f"{self.metric_dir}/skewness_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.skweness_4, nesting=2)
            save_to_json(self.grad_correlation_4, f"{self.metric_dir}/grad_correlation_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.grad_correlation_4, nesting=2)
            torch.save(self.difference_heatmaps_4, f'{self.metric_dir}/difference_heatmaps_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.difference_heatmaps_4, nesting=1)
            torch.save(self.grad_real_A_4, f'{self.metric_dir}/grad_real_A_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_real_A_4, nesting=1)
            torch.save(self.grad_fake_B_4, f'{self.metric_dir}/grad_fake_B_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.grad_fake_B_4, nesting=1)
            torch.save(self.gradient_difference_4, f'{self.metric_dir}/gradient_difference_{self.opt.test}_epoch{epoch}.pth')
            empty_Dictionary(self.gradient_difference_4, nesting=1)

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []

    def save_raps_per_patient(self, epoch):
        if self.opt.test == "test_3":
            save_to_json(self.raps_data_3, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.raps_data_3, nesting=2)
        elif self.opt.test == "elcap_complete":
            save_to_json(self.raps_data_4, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
            empty_Dictionary(self.raps_data_4, nesting=2)

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
        else:
            raise ("NotImplemented")

    def save_test_images(self, epoch):
        standardized_test = torch.cat(self.standardized_test_buffer, dim=0)
        not_standardized_test = torch.cat(self.not_standardized_test_buffer, dim=0)
        torch.save(standardized_test, f'{self.test_dir}/standardized_{self.opt.test}_epoch{epoch}.pth')
        torch.save(not_standardized_test, f'{self.test_dir}/not_standardized_{self.opt.test}_epoch{epoch}.pth')
        self.standardized_test_buffer = []
        self.not_standardized_test_buffer = []

    def save_noise_metrics(self, epoch):
        # torch.save(self.kurtosis, f'{self.metric_dir}/kurtosis_{self.opt.test}_epoch{epoch}.pth')
        torch.save(self.skweness, f'{self.metric_dir}/skweness_{self.opt.test}_epoch{epoch}.pth')
        # torch.save(self.shannon_entropy, f'{self.metric_dir}/shannon_entropy_{self.opt.test}_epoch{epoch}.pth')

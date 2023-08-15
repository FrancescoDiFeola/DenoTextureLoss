from .base_model import BaseModel, OrderedDict
from . import networks
import os
import itertools
from .vgg import VGG
from util.util import tensor2im2, save_list_to_csv, save_json
import pyiqa
from loss_functions.attention import Self_Attn
from metrics.FID import *
from metrics.mse_psnr_ssim_vif import *
from loss_functions.texture_loss import texture_loss
from loss_functions.perceptual_loss import perceptual_similarity_loss
from models.networks import init_net
from piq import brisque


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--experiment_name', type=str, default="default", help='experiment name')
        parser.add_argument('--image_folder', type=str, default=None, help='folder to save images during training')
        parser.add_argument('--metric_folder', type=str, default=None, help='folder to save metrics')
        parser.add_argument('--loss_folder', type=str, default=None, help='folder to save losses')
        parser.add_argument('--test_folder', type=str, default=None, help='folder to save test images')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_texture', type=float, default=0.001, help='use texture loss.')
            parser.add_argument('--texture_criterion', type=str, default="max", help='texture loss criterion.')
            parser.add_argument('--texture_offsets', type=str, default="all", help='texture offsets.')
            parser.add_argument('--vgg_pretrained', type=str, default=True, help='pretraining flag.')
            parser.add_argument('--vgg_model_path', type=str, default=None, help='finetuned vgg model path.')
            parser.add_argument('--lambda_perceptual', type=float, default=0.1, help='use perceptual loss.')
            parser.add_argument('--perceptual_layers', type=str, default='all', help='choose the perceptual layers.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if opt.experiment_name.find('texture') != -1:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'texture']
        elif opt.experiment_name.find('perceptual') != -1:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'perceptual']
        elif opt.experiment_name.find('baseline') != -1:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

        # dictionary to store training loss
        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()

        self.test_visual_names = ['real_A', 'fake_B', 'real_B']
        self.metric_names = ['psnr', 'mse', 'ssim', 'vif', 'NIQE', 'FID', 'brisque']
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.loss_dir = os.path.join(self.web_dir, f'{opt.loss_folder}')
        self.metric_dir = os.path.join(self.web_dir, f'{opt.metric_folder}')
        self.test_dir = os.path.join(self.web_dir, f'{opt.test_folder}')

        self.metrics_eval = OrderedDict()
        for key in self.metric_names:
            self.metrics_eval[key] = list()

        self.avg_metrics_test_1 = OrderedDict()
        self.avg_metrics_test_2 = OrderedDict()
        self.avg_metrics_test_3 = OrderedDict()

        for key in self.metric_names:
            self.avg_metrics_test_1[key] = OrderedDict()
            self.avg_metrics_test_2[key] = OrderedDict()
            self.avg_metrics_test_3[key] = OrderedDict()

            self.avg_metrics_test_1[key]['mean'] = list()
            self.avg_metrics_test_1[key]['std'] = list()
            self.avg_metrics_test_2[key]['mean'] = list()
            self.avg_metrics_test_2[key]['std'] = list()
            self.avg_metrics_test_3[key]['mean'] = list()
            self.avg_metrics_test_3[key]['std'] = list()

        self.fid_object = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64)

        self.real_test_buffer = []
        self.fake_test_buffer = []
        self.raps = list()

        # NIQE metric
        self.niqe = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionTexture = torch.nn.L1Loss(reduction='none')
            if opt.texture_criterion == 'attention':
                self.attention = init_net(Self_Attn(1, 'relu'))
                self.weight = list()
                self.attention_B = list()
                # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.attention.parameters()), lr=opt.lr,
                                                    betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if opt.lambda_perceptual > 0.0:
                if opt.vgg_pretrained == True:
                    self.vgg = VGG().to(int(opt.gpu_ids[1]))
                else:
                    self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned',
                                   saved_weights_path=opt.vgg_model_path).to(int(opt.gpu_ids[1]))

            self.index_texture = list()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        if self.opt.dataset_mode == "LIDC_IDRI":
            self.real_A = input['img']
            self.image_paths = input['im_paths']
        else:
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def test(self, idx):

        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)
            self.compute_metrics(idx)
            self.track_metrics()

    def compute_metrics(self, idx):
        if self.opt.test == "test_3":
            # NIQE
            fake_B_3channels = self.fake_B.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_B_3channels).item()
            self.raps.append(azimuthalAverage(np.squeeze(self.fake_B[0, 0, :, :].cpu().detach().numpy())).tolist())
            self.brisque = 0
            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0
            self.fid = 0
        elif self.opt.test == "elcap_complete":
            # NIQE
            fake_B_3channels = self.fake_B.expand(-1, 3, -1, -1)
            self.NIQE = self.niqe(fake_B_3channels).item()
            self.raps.append(azimuthalAverage(np.squeeze(self.fake_B[0, 0, :, :].cpu().detach().numpy())).tolist())
            self.brisque = 0
            self.mse = 0
            self.psnr = 0
            self.ssim = 0
            self.vif = 0
            self.fid = 0
        else:
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
            # FID
            self.fake_test_buffer.append(self.fake_B)
            self.real_test_buffer.append(self.real_B)

    '''def compute_fid(self, idx):
        if idx == self.opt.dataset_len - 1:
            self.real_test_buffer = torch.cat(self.real_test_buffer)
            self.fake_test_buffer = torch.cat(self.fake_test_buffer)
            fid_index = calculate_frechet(self.real_test_buffer, self.fake_test_buffer, self.inception_model)
            self.metrics_eval['FID'].append(fid_index)
            self.real_test_buffer = []
            self.fake_test_buffer = []
        else:
            pass'''

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        # Texture loss
        lambda_texture = self.opt.lambda_texture
        if lambda_texture > 0:

            if self.opt.texture_criterion == 'attention':
                loss_texture, map_B, weight_B = texture_loss(self.fake_B, self.real_B, self.criterionTexture, self.opt,
                                                             self.attention)  # , map_B, weight_B
                self.loss_texture = loss_texture

                self.weight.append(weight_B.item())
                # self.attention_B.append(map_B)

            elif self.opt.texture_criterion == 'max':
                loss_texture, delta_grids_B, criterion_texture_B = texture_loss(self.fake_B, self.real_B,
                                                                                self.criterionTexture, self.opt)

                # save the index of the maximum texture descriptor for each image in the batch
                for i in range(self.opt.batch_size):
                    self.index_texture.append(torch.nonzero(delta_grids_B == criterion_texture_B[i]).squeeze())

                # compute the loss function by averaging over the batch
                self.loss_texture = loss_texture * lambda_texture

            elif self.opt.texture_criterion == 'average':
                loss_texture = texture_loss(self.fake_B, self.real_B, self.criterionTexture, self.opt)

                self.loss_texture = loss_texture * lambda_texture

            elif self.opt.texture_criterion == 'Frobenius':
                loss_texture = texture_loss(self.fake_B, self.real_B, self.criterionTexture, self.opt)

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

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_texture + self.loss_perceptual
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # update G's weights

    def save_texture_indexes(self):
        save_list_to_csv(self.index_texture, f'{self.loss_dir}/idx_texture.csv')

    def save_attention_maps(self):
        np.save(f"{self.loss_dir}/attention_B.npy", np.array(self.attention_B))

    def save_attention_weights(self):
        np.save(f"{self.loss_dir}/weight.npy", np.array(self.weight))

    def save_list_images(self, epoch):
        real_buffer = torch.cat(self.real_test_buffer, dim=0)
        fake_buffer = torch.cat(self.fake_test_buffer, dim=0)

        fid_score = self.fid_object.compute_fid(fake_buffer, real_buffer, self.opt.dataset_len)
        self.metrics_eval['FID'].append(fid_score)

        # torch.save(real_buffer, f'{self.test_dir}/real_buffer_{self.opt.test}_epoch{epoch}.pth')
        # torch.save(fake_buffer, f'{self.test_dir}/fake_buffer_{self.opt.test}_epoch{epoch}.pth')

        self.real_test_buffer = []
        self.fake_test_buffer = []

    def save_raps(self, epoch):
        save_json(self.raps, f"{self.metric_dir}/raps_{self.opt.test}_epoch{epoch}")
        self.raps = []
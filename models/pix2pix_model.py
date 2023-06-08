import torch
from .base_model import BaseModel, OrderedDict
from . import networks
import os
from .vgg import VGG
from util.util import tensor2im2, save_list_to_csv
import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from math import log10
from skimage.metrics import structural_similarity
from sewar.full_ref import vifp
from surgeon_pytorch import Inspect

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
        parser.set_defaults(norm='batch', netG='unet_256')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
            parser.add_argument('--experiment_name', type=str, default="default", help='experiment name')
            parser.add_argument('--image_folder', type=str, default="images",
                                help='folder to save images during training')
            parser.add_argument('--metric_folder', type=str, default="metrics", help='folder to save metrics')
            parser.add_argument('--loss_folder', type=str, default="losses", help='folder to save losses')
            parser.add_argument('--test_folder', type=str, default="test", help='folder to save test images')
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
        self.metric_names = ['psnr', 'ssim', 'vif']
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.loss_dir = os.path.join(self.web_dir, f'{opt.loss_folder}')

        self.metrics_eval = OrderedDict()
        for key in self.metric_names:
            self.metrics_eval[key] = list()
        self.avg_metrics = OrderedDict()

        for key in self.metric_names:
            self.avg_metrics[key] = OrderedDict()
            self.avg_metrics[key]['mean'] = list()
            self.avg_metrics[key]['std'] = list()

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
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
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if opt.lambda_perceptual > 0.0:
                if opt.vgg_pretrained == True:
                    self.vgg = VGG().to(int(opt.gpu_ids[1]))
                else:
                    self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned', saved_weights_path=opt.vgg_model_path).to(int(opt.gpu_ids[1]))
            self.index_texture_A = list()
            self.index_texture_B = list()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def test(self):
        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)
            self.compute_metrics()
            self.track_metrics()

    def compute_metrics(self):

        x = tensor2im2(self.real_B)
        y = tensor2im2(self.fake_B)

        # PSNR
        mse = np.square(np.subtract(x, y)).mean()
        if mse == 0:  # MSE is zero means no noise is present in the signal. Therefore, PSNR have no importance.
            self.psnr = 100
        max_pixel = 1

        self.psnr = 10 * log10((max_pixel ** 2) / mse)

        # SSIM
        self.ssim = structural_similarity(x, y, data_range=2)

        # VIF
        self.vif = vifp(x, y)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
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

        lambda_texture = self.opt.lambda_texture
        # Texture loss
        if lambda_texture > 0:

            self.textures_real_B = self.texture_extractor(self.real_B)
            self.textures_fake_B = self.texture_extractor(self.fake_B)
            criterion_texture_B = self.criterionTexture(self.textures_fake_B, self.textures_real_B)

            if self.opt.texture_criterion == 'max':
                loss_cycle_texture_B = torch.max(criterion_texture_B)
                self.index_texture_B.append(torch.where(criterion_texture_B == loss_cycle_texture_B)[2:])
                self.loss_texture = loss_cycle_texture_B * lambda_texture

            elif self.opt.texture_criterion == 'normalized':
                normalizing_factor_real_B = torch.max(self.textures_real_B)
                loss_cycle_texture_B = torch.sum(criterion_texture_B) / normalizing_factor_real_B
                self.loss_texture = loss_cycle_texture_B * 1

            elif self.opt.texture_criterion == 'average':
                self.loss_texture = torch.mean(criterion_texture_B) * lambda_texture
            else:
                raise NotImplementedError
        else:
            self.loss_cycle_texture = 0

        lambda_perceptual = self.opt.lambda_perceptual
        if lambda_perceptual > 0:
            loss_perceptual_B = self.perceptual_similarity_loss()
            self.loss_perceptual_B = loss_perceptual_B * lambda_perceptual
        else:
            self.loss_perceptual_B = 0

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

    def texture_extractor(self, x):

        x = cv2.normalize(x.detach().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F).astype(np.uint8)

        if self.opt.texture_offsets == 'all':
            shape = (x.shape[0], 1, 4, 4)  # 4,4
            spatial_offset = [1, 3, 5, 7]
            angular_offset = [0, 45, 90, 135]
        elif self.opt.texture_offsets == '5':
            shape = (x.shape[0], 1, 1, 4)  # 4,4
            spatial_offset = [5]
            angular_offset = [0, 45, 90, 135]

        texture_matrix = torch.empty(shape)

        for i in range(0, x.shape[0]):
            for idx_d, d in enumerate(spatial_offset):
                for idx_theta, theta in enumerate(angular_offset):
                    texture_d_theta = graycoprops(
                        graycomatrix(x[i, 0, :, :], distances=[d], angles=[theta], levels=256, symmetric=True,
                                     normed=True), "contrast")[0][0]
                    texture_matrix[i, 0, idx_d - 1, idx_theta - 1] = texture_d_theta

        return texture_matrix

    def perceptual_similarity_loss(self):


        fake_B = self.rec_B.expand(-1, 3, -1, -1)
        real_B = self.real_B.expand(-1, 3, -1, -1)

        f1_real_B, f2_real_B, f3_real_B, f4_real_B, f5_real_B = self.vgg_16_pretrained(real_B)

        f1_fake_B, f2_fake_B, f3_fake_B, f4_fake_B, f5_fake_B = self.vgg_16_pretrained(fake_B)

        if self.opt.perceptual_layers == 'all':
            m1_B = torch.mean(torch.mean((f1_real_B - f1_fake_B) ** 2))
            m2_B = torch.mean(torch.mean((f2_real_B - f2_fake_B) ** 2))
            m3_B = torch.mean(torch.mean((f3_real_B - f3_fake_B) ** 2))
            m4_B = torch.mean(torch.mean((f4_real_B - f4_fake_B) ** 2))
            m5_B = torch.mean(torch.mean((f5_real_B - f5_fake_B) ** 2))
            perceptual_loss_B = m1_B + m2_B + m3_B + m4_B + m5_B
        elif self.opt.perceptual_layers == '1-2':
            m1_B = torch.mean(torch.mean((f1_real_B - f1_fake_B) ** 2))
            m2_B = torch.mean(torch.mean((f2_real_B - f2_fake_B) ** 2))
            perceptual_loss_B = m1_B + m2_B
        elif self.opt.perceptual_layers == '4-5':
            m4_B = torch.mean(torch.mean((f4_real_B - f4_fake_B) ** 2))
            m5_B = torch.mean(torch.mean((f5_real_B - f5_fake_B) ** 2))
            perceptual_loss_B = m4_B + m5_B
        elif self.opt.perceptual_layers == '2-4':
            m2_B = torch.mean(torch.mean((f2_real_B - f2_fake_B) ** 2))
            m4_B = torch.mean(torch.mean((f4_real_B - f4_fake_B) ** 2))
            perceptual_loss_B = m2_B + m4_B

        return  perceptual_loss_B

    # EXTRACT THE POOLING LAYERS FROM THE PRETRAINED VGG-16
    def vgg_16_pretrained(self, image):
        wrapped_model = Inspect(self.vgg, layer=['features.4', 'features.9', 'features.16', 'features.23', 'features.30'])
        _, [p1, p2, p3, p4, p5] = wrapped_model(image)

        return p1, p2, p3, p4, p5

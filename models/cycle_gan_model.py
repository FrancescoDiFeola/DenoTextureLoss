import torch
import itertools
from util.image_pool import ImagePool
from util.util import tensor2im, empty_dictionary, tensor2im2, save_list_to_csv, load_saved_weights
from .base_model import BaseModel, OrderedDict
from . import networks
import numpy as np
from math import log10
from skimage.metrics import structural_similarity
from sewar.full_ref import vifp
import cv2
from skimage.feature import graycomatrix, graycoprops
import torchvision.models as models
from surgeon_pytorch import Inspect
import os
from .vgg import VGG

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--experiment_name', type=str, default="default", help='experiment name')
        parser.add_argument('--image_folder', type=str, default="images", help='folder to save images during training')
        parser.add_argument('--metric_folder', type=str, default="metrics", help='folder to save metrics')
        parser.add_argument('--loss_folder', type=str, default="losses", help='folder to save losses')
        parser.add_argument('--test_folder', type=str, default="test", help='folder to save test images')
        if is_train:
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
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if opt.experiment_name.find('texture') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'cycle_texture_A', 'D_B', 'G_B', 'cycle_B', 'idt_B',
                                    'cycle_texture_B']
        elif opt.experiment_name.find('perceptual') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'perceptual_A', 'D_B', 'G_B', 'cycle_B', 'idt_B',
                               'perceptual_B']
        elif opt.experiment_name.find('baseline') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # dictionary to store training loss
        self.error_store = OrderedDict()
        for key in self.loss_names:
            self.error_store[key] = list()

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        #####
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

        #####
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionTexture = torch.nn.L1Loss(reduction='none')
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
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

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def test(self):
        with torch.no_grad():
            self.fake_B = self.netG_A(self.real_A)
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

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
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
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        lambda_texture = self.opt.lambda_texture
        # Texture loss
        if lambda_texture > 0:
            self.textures_real_A = self.texture_extractor(self.real_A)
            self.textures_rec_A = self.texture_extractor(self.rec_A)
            self.textures_real_B = self.texture_extractor(self.real_B)
            self.textures_rec_B = self.texture_extractor(self.rec_B)
            criterion_texture_A = self.criterionTexture(self.textures_rec_A, self.textures_real_A)
            criterion_texture_B = self.criterionTexture(self.textures_rec_B, self.textures_real_B)

            if self.opt.texture_criterion == 'max':
                loss_cycle_texture_A = torch.max(criterion_texture_A)
                loss_cycle_texture_B = torch.max(criterion_texture_B)
                self.index_texture_A.append(torch.where(criterion_texture_A == loss_cycle_texture_A)[2:])
                self.index_texture_B.append(torch.where(criterion_texture_B == loss_cycle_texture_B)[2:])
                self.loss_cycle_texture_A = loss_cycle_texture_A * lambda_texture
                self.loss_cycle_texture_B = loss_cycle_texture_B * lambda_texture

            elif self.opt.texture_criterion == 'normalized':
                normalizing_factor_real_A = torch.max(self.textures_real_A)
                normalizing_factor_real_B = torch.max(self.textures_real_B)
                loss_cycle_texture_A = torch.sum(criterion_texture_A)/normalizing_factor_real_A
                loss_cycle_texture_B = torch.sum(criterion_texture_B)/normalizing_factor_real_B
                self.loss_cycle_texture_A = loss_cycle_texture_A * 1
                self.loss_cycle_texture_B = loss_cycle_texture_B * 1

            elif self.opt.texture_criterion == 'average':
                self.loss_cycle_texture_A = torch.mean(criterion_texture_A) * lambda_texture
                self.loss_cycle_texture_B = torch.mean(criterion_texture_B) * lambda_texture
            else:
                raise NotImplementedError
        else:
            self.loss_cycle_texture_A = 0
            self.loss_cycle_texture_B = 0

        lambda_perceptual = self.opt.lambda_perceptual
        if lambda_perceptual > 0:
            loss_perceptual_A, loss_perceptual_B = self.perceptual_similarity_loss()
            self.loss_perceptual_A = loss_perceptual_A * lambda_perceptual
            self.loss_perceptual_B = loss_perceptual_B * lambda_perceptual
        else:
            self.loss_perceptual_A = 0
            self.loss_perceptual_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_cycle_texture_A + self.loss_cycle_texture_B + self.loss_perceptual_A + self.loss_perceptual_B

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

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

        rec_A = self.rec_A.expand(-1, 3, -1, -1)
        rec_B = self.rec_B.expand(-1, 3, -1, -1)
        real_A = self.real_A.expand(-1, 3, -1, -1)
        real_B = self.real_B.expand(-1, 3, -1, -1)

        f1_real_A, f2_real_A, f3_real_A, f4_real_A, f5_real_A = self.vgg_16_pretrained(real_A)  # extract features from vgg
        f1_real_B, f2_real_B, f3_real_B, f4_real_B, f5_real_B = self.vgg_16_pretrained(real_B)

        f1_rec_A, f2_rec_A, f3_rec_A, f4_rec_A, f5_rec_A = self.vgg_16_pretrained(rec_A)  # extract features from vgg (1st,2nd, 3rd, 4th, 5th pool)
        f1_rec_B, f2_rec_B, f3_rec_B, f4_rec_B, f5_rec_B = self.vgg_16_pretrained(rec_B)

        if self.opt.perceptual_layers == 'all':
            m1_A = torch.mean(torch.mean((f1_real_A - f1_rec_A) ** 2))  # mse difference
            m2_A = torch.mean(torch.mean((f2_real_A - f2_rec_A) ** 2))
            m3_A = torch.mean(torch.mean((f3_real_A - f3_rec_A) ** 2))
            m4_A = torch.mean(torch.mean((f4_real_A - f4_rec_A) ** 2))
            m5_A = torch.mean(torch.mean((f5_real_A - f5_rec_A) ** 2))

            m1_B = torch.mean(torch.mean((f1_real_B - f1_rec_B) ** 2))
            m2_B = torch.mean(torch.mean((f2_real_B - f2_rec_B) ** 2))
            m3_B = torch.mean(torch.mean((f3_real_B - f3_rec_B) ** 2))
            m4_B = torch.mean(torch.mean((f4_real_B - f4_rec_B) ** 2))
            m5_B = torch.mean(torch.mean((f5_real_B - f5_rec_B) ** 2))

            perceptual_loss_A = m1_A + m2_A + m3_A + m4_A + m5_A
            perceptual_loss_B = m1_B + m2_B + m3_B + m4_B + m5_B
        elif self.opt.perceptual_layers == '1-2':
            m1_A = torch.mean(torch.mean((f1_real_A - f1_rec_A) ** 2))  # mse difference
            m2_A = torch.mean(torch.mean((f2_real_A - f2_rec_A) ** 2))

            m1_B = torch.mean(torch.mean((f1_real_B - f1_rec_B) ** 2))
            m2_B = torch.mean(torch.mean((f2_real_B - f2_rec_B) ** 2))
            perceptual_loss_A = m1_A + m2_A
            perceptual_loss_B = m1_B + m2_B
        elif self.opt.perceptual_layers == '4-5':
            m4_A = torch.mean(torch.mean((f4_real_A - f4_rec_A) ** 2))
            m5_A = torch.mean(torch.mean((f5_real_A - f5_rec_A) ** 2))

            m4_B = torch.mean(torch.mean((f4_real_B - f4_rec_B) ** 2))
            m5_B = torch.mean(torch.mean((f5_real_B - f5_rec_B) ** 2))

            perceptual_loss_A = m4_A + m5_A
            perceptual_loss_B = m4_B + m5_B
        elif self.opt.perceptual_layers == '2-4':

            m2_A = torch.mean(torch.mean((f2_real_A - f2_rec_A) ** 2))
            m4_A = torch.mean(torch.mean((f4_real_A - f4_rec_A) ** 2))

            m2_B = torch.mean(torch.mean((f2_real_B - f2_rec_B) ** 2))
            m4_B = torch.mean(torch.mean((f4_real_B - f4_rec_B) ** 2))

            perceptual_loss_A = m2_A + m4_A
            perceptual_loss_B = m2_B + m4_B

        return perceptual_loss_A, perceptual_loss_B

    # EXTRACT THE POOLING LAYERS FROM THE PRETRAINED VGG-16
    def vgg_16_pretrained(self, image):
        wrapped_model = Inspect(self.vgg, layer=['features.4', 'features.9', 'features.16', 'features.23', 'features.30'])
        _, [p1, p2, p3, p4, p5] = wrapped_model(image)

        return p1, p2, p3, p4, p5

    def save_texture_indexes(self):
        save_list_to_csv(self.index_texture_A, f'{self.loss_dir}/idx_texture_A.csv')
        save_list_to_csv(self.index_texture_B, f'{self.loss_dir}/idx_texture_B.csv')
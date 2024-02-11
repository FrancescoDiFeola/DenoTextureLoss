import itertools
from util.image_pool import ImagePool
from util.util import *
from .base_model import BaseModel, OrderedDict
from data.storage import *
from . import networks
import os
import pyiqa
from .vgg import VGG
from loss_functions.attention import Self_Attn
from metrics.FID import *
from loss_functions.perceptual_loss import perceptual_similarity_loss
# from loss_functions.texture_loss import texture_loss
from loss_functions.glcm_soft_einsum_ import _texture_loss_d5, _GridExtractor_d5, _texture_loss, _GridExtractor
from metrics.mse_psnr_ssim_vif import *
from models.networks import init_net
from piq import brisque
from scipy.stats import skew


class AttentionGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--experiment_name', type=str, default="default", help='experiment name')
        parser.add_argument('--image_folder', type=str, default=None, help='folder to save images during training')
        parser.add_argument('--metric_folder', type=str, default=None, help='folder to save metrics')
        parser.add_argument('--loss_folder', type=str, default=None, help='folder to save losses')
        parser.add_argument('--test_folder', type=str, default=None, help='folder to save test images')
        parser.add_argument('--test', type=str, default="test_1", help='folder to save test images')
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
            parser.add_argument('--attention_loss', type=bool, default=False, help='choose to use attention mechanism for loss computation.')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if opt.experiment_name.find('texture') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'cycle_texture_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'cycle_texture_B']
        elif opt.experiment_name.find('perceptual') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'perceptual_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'perceptual_B']
        elif opt.experiment_name.find('baseline') != -1:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'o1_b', 'o2_b', 'o3_b', 'o4_b', 'o5_b', 'o6_b', 'o7_b', 'o8_b', 'o9_b', 'o10_b',
                          'a1_b', 'a2_b', 'a3_b', 'a4_b', 'a5_b', 'a6_b', 'a7_b', 'a8_b', 'a9_b', 'a10_b', 'i1_b', 'i2_b', 'i3_b', 'i4_b', 'i5_b',
                          'i6_b', 'i7_b', 'i8_b', 'i9_b']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'o1_a', 'o2_a', 'o3_a', 'o4_a', 'o5_a', 'o6_a', 'o7_a', 'o8_a', 'o9_a', 'o10_a',
                          'a1_a', 'a2_a', 'a3_a', 'a4_a', 'a5_a', 'a6_a', 'a7_a', 'a8_a', 'a9_a', 'a10_a', 'i1_a', 'i2_a', 'i3_a', 'i4_a', 'i5_a',
                          'i6_a', 'i7_a', 'i8_a', 'i9_a']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        if self.opt.saveDisk:
            self.visual_names = ['real_A', 'fake_B', 'a10_b', 'real_B', 'fake_A', 'a10_a']
        else:
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.

        self.metric_names = ['psnr', 'mse', 'ssim', 'vif', 'NIQE', 'FID', 'brisque']
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.loss_dir = os.path.join(self.web_dir, f'{opt.loss_folder}')
        self.test_dir = os.path.join(self.web_dir, f'{opt.test_folder}')
        self.metric_dir = os.path.join(self.web_dir, f'{opt.metric_folder}')

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
        self.fid_object_1 = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64, pretrained=True)
        self.fid_object_2 = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64, pretrained=False)
        # self.fake_test_buffer = []
        self.real_buffer_2 = init_image_buffer(test_2_ids)
        self.fake_buffer_2 = init_image_buffer(test_2_ids)

        self.raps = 0

        # NIQE
        self.niqe = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)
        # self.brisque = pyiqa.create_metric('brisque', device=torch.device('cpu'), as_loss=False)

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'our', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, [0, 1, 2, 3])
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, 'our', opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, [0, 1, 2, 3])
        self.skweness = []

        self.grid_extractor = _GridExtractor()
        # Wrap the texture_extractor in DataParallel if you have multiple GPUs
        if torch.cuda.device_count() > 1:
            self.grid_extractor = nn.DataParallel(self.grid_extractor, device_ids=[0, 1, 2, 3])

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, [0, 1, 2, 3])
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, [0, 1, 2, 3])

            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss_functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionTexture = torch.nn.L1Loss(reduction='none')  # operator

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.texture_criterion == 'attention':
                self.attention = init_net(Self_Attn(1, 'relu')).to(self.device)

                # self.optimizer_att = torch.optim.Adam(self.attention.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.weight_A = list()
                self.weight_B = list()
                self.attention_A = list()
                self.attention_B = list()

                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.attention.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if opt.lambda_perceptual > 0.0:
                if opt.vgg_pretrained == True:
                    self.vgg = VGG().eval().to("cuda:1")
                else:
                    self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned',
                                   saved_weights_path=opt.vgg_model_path)
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
    self.fake_B, self.o1_b, self.o2_b, self.o3_b, self.o4_b, self.o5_b, self.o6_b, self.o7_b, self.o8_b, self.o9_b, self.o10_b, \
    self.a1_b, self.a2_b, self.a3_b, self.a4_b, self.a5_b, self.a6_b, self.a7_b, self.a8_b, self.a9_b, self.a10_b, \
    self.i1_b, self.i2_b, self.i3_b, self.i4_b, self.i5_b, self.i6_b, self.i7_b, self.i8_b, self.i9_b = self.netG_A(self.real_A)  # G_A(A)
    self.rec_A, _, _, _, _, _, _, _, _, _, _, \
    _, _, _, _, _, _, _, _, _, _, \
    _, _, _, _, _, _, _, _, _ = self.netG_B(self.fake_B)  # G_B(G_A(A))
    self.fake_A, self.o1_a, self.o2_a, self.o3_a, self.o4_a, self.o5_a, self.o6_a, self.o7_a, self.o8_a, self.o9_a, self.o10_a, \
    self.a1_a, self.a2_a, self.a3_a, self.a4_a, self.a5_a, self.a6_a, self.a7_a, self.a8_a, self.a9_a, self.a10_a, \
    self.i1_a, self.i2_a, self.i3_a, self.i4_a, self.i5_a, self.i6_a, self.i7_a, self.i8_a, self.i9_a = self.netG_B(self.real_B)  # G_B(B)
    self.rec_B, _, _, _, _, _, _, _, _, _, _, \
    _, _, _, _, _, _, _, _, _, _, \
    _, _, _, _, _, _, _, _, _ = self.netG_A(self.fake_A)  # G_A(G_B(B))


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
        self.idt_A, _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _ = self.netG_A(self.real_B)
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        self.idt_B, _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _, _, \
        _, _, _, _, _, _, _, _, _ = self.netG_B(self.real_A)
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
    else:
        self.loss_idt_A = 0
        self.loss_idt_B = 0

    lambda_texture = self.opt.lambda_texture

    # Texture loss
    if lambda_texture > 0:
        if self.opt.texture_criterion == 'attention':
            loss_texture_A, map_A, weight_A = _texture_loss(self.rec_A, self.real_A, self.opt, self.grid_extractor,
                                                            self.attention)
            loss_texture_B, map_B, weight_B = _texture_loss(self.rec_B, self.real_B, self.opt, self.grid_extractor,
                                                            self.attention)

            self.loss_cycle_texture_A = loss_texture_A
            self.loss_cycle_texture_B = loss_texture_B
            self.weight_A.append(weight_A.item())
            self.weight_B.append(weight_B.item())
            self.attention_A.append(map_A.detach().clone().cpu().numpy())
            self.attention_B.append(map_B.detach().clone().cpu().numpy())

        elif self.opt.texture_criterion == 'max':
            loss_texture_A = _texture_loss_d5(self.rec_A, self.real_A, self.opt, self.grid_extractor)
            loss_texture_B = _texture_loss_d5(self.rec_B, self.real_B, self.opt, self.grid_extractor)

            # save the index of the maximum texture descriptor for each image in the batch
            '''for i in range(self.opt.batch_size):
                self.index_texture_A.append(torch.nonzero(delta_grids_A == criterion_texture_A[i]).squeeze())
                self.index_texture_B.append(torch.nonzero(delta_grids_B == criterion_texture_B[i]).squeeze())'''

            # compute the loss function by averaging over the batch
            self.loss_cycle_texture_A = loss_texture_A * lambda_texture
            self.loss_cycle_texture_B = loss_texture_B * lambda_texture

        elif self.opt.texture_criterion == 'average':
            loss_texture_A = _texture_loss_d5(self.rec_A, self.real_A, self.opt, self.grid_extractor)
            loss_texture_B = _texture_loss_d5(self.rec_B, self.real_B, self.opt, self.grid_extractor)

            self.loss_cycle_texture_A = loss_texture_A * lambda_texture
            self.loss_cycle_texture_B = loss_texture_B * lambda_texture

        elif self.opt.texture_criterion == 'Frobenius':
            loss_texture_A = _texture_loss_d5(self.rec_A, self.real_A, self.opt, self.grid_extractor)
            loss_texture_B = _texture_loss_d5(self.rec_B, self.real_B, self.opt, self.grid_extractor)

            self.loss_cycle_texture_A = loss_texture_A * lambda_texture
            self.loss_cycle_texture_B = loss_texture_B * lambda_texture
        else:
            raise NotImplementedError
    else:
        self.loss_cycle_texture_A = 0
        self.loss_cycle_texture_B = 0

    # Perceptual loss
    lambda_perceptual = self.opt.lambda_perceptual
    if lambda_perceptual > 0:
        loss_perceptual_A = perceptual_similarity_loss(self.rec_A, self.real_A, self.vgg,
                                                       self.opt.perceptual_layers)
        loss_perceptual_B = perceptual_similarity_loss(self.rec_B, self.real_B, self.vgg,
                                                       self.opt.perceptual_layers)
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


def save_texture_indexes(self):
    save_list_to_csv(self.index_texture_A, f'{self.loss_dir}/idx_texture_A.csv')
    save_list_to_csv(self.index_texture_B, f'{self.loss_dir}/idx_texture_B.csv')


def save_attention_maps(self):
    np.save(f"{self.loss_dir}/attention_A.npy", np.array(self.attention_A))
    np.save(f"{self.loss_dir}/attention_B.npy", np.array(self.attention_B))


def save_attention_weights(self):
    np.save(f"{self.loss_dir}/weight_A.npy", np.array(self.weight_A))
    np.save(f"{self.loss_dir}/weight_B.npy", np.array(self.weight_B))


def fid_compute(self):
    for key in self.real_buffer_2.keys():
        real_buffer = torch.cat(self.real_buffer_2[key], dim=0)
        fake_buffer = torch.cat(self.fake_buffer_2[key], dim=0)

        fid_score_1 = self.fid_object_1.compute_fid(fake_buffer, real_buffer, self.opt.dataset_len)
        fid_score_2 = self.fid_object_2.compute_fid(fake_buffer, real_buffer, self.opt.dataset_len)

        self.metrics_data_2[key]['FID_ImNet'].append(fid_score_1)
        self.metrics_data_2[key]['FID_random'].append(fid_score_2)

    empty_Dictionary(self.real_buffer_2, nesting=1)
    empty_Dictionary(self.fake_buffer_2, nesting=1)


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


def save_noise_metrics(self, epoch):
    # torch.save(self.kurtosis, f'{self.metric_dir}/kurtosis_{self.opt.test}_epoch{epoch}.pth')
    torch.save(self.skweness, f'{self.metric_dir}/skweness_{self.opt.test}_epoch{epoch}.pth')
    # torch.save(self.shannon_entropy, f'{self.metric_dir}/shannon_entropy_{self.opt.test}_epoch{epoch}.pth')
    self.optimizer_D.step()  # update D_A and D_B's weights
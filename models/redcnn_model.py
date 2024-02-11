from models.redcnn_networks import *
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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class REDCNNModel(BaseModel):


    def __init__(self, opt):
        """Initialize the UNIT class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        super().__init__(opt)
        self.criterion_mse = torch.nn.MSELoss()
        self.criterionTexture = torch.nn.L1Loss(reduction='none')

        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.Tensor = torch.FloatTensor if str(self.device).find("cpu") != -1 else torch.Tensor

        # Initialize generator and discriminator
        netG = Generator(in_channels=1, out_channels=opt.num_channels, num_layers=opt.num_layers).to(self.device)
        self.optimizer = torch.optim.Adam(netG.parameters(), opt.init_lr)

        self.netG = nn.DataParallel(netG, device_ids=[0, 1])  # .to(self.device)

        self.criterion_mse.to(self.device)
        self.criterionTexture.to(self.device)

        # Initialize weights
        self.netG.apply(weights_init_normal)

        # Optimizers
        if opt.texture_criterion == 'attention':
            self.attention = init_net(Self_Attn(1, 'relu')).to(self.device)
            self.weight = list()
            self.attention_B = list()
            self.optimizer = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.attention.parameters()), lr=opt.init_lr)
            self.model_names = ['netG', "attention"]
        else:
            self.optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.init_lr)
            self.model_names = ['netG']

        # Learning rate update schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

        if opt.lambda_perceptual > 0.0:
            if opt.vgg_pretrained:
                self.vgg = VGG().eval().to("cuda:1")
            else:
                self.vgg = VGG(num_classes=4, pretrained=False, init_type='finetuned', saved_weights_path=opt.vgg_model_path).eval().to("cuda:1")

        # dictionary to store training loss
        if opt.lambda_texture > 0:
            self.loss_names = ['G_mse', 'texture']
        elif opt.lambda_perceptual > 0:
            self.loss_names = ['G_mse', 'perceptual']
        else:
            self.loss_names = ['G_mse']

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

        # self.real_test_buffer = []
        # self.fake_test_buffer = []
        self.real_buffer_2 = []  # init_image_buffer(test_2_ids)
        self.fake_buffer_2 = []  # init_image_buffer(test_2_ids)

        self.raps = 0
        self.skweness = []

        self.grid_extractor = _GridExtractor()
        # Wrap the texture_extractor in DataParallel if you have multiple GPUs
        if torch.cuda.device_count() > 1:
            self.grid_extractor = nn.DataParallel(self.grid_extractor, device_ids=[0, 1])

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

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        if self.opt.dataset_mode == "LIDC_IDRI":
            self.real_A = input['img'].to(self.device)
            self.id = input['patient'][0]
            # self.image_paths = input['im_paths']
        else:
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.id = input['patient'][0]
            # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def test(self):

        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)
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

            self.fake_buffer_2.append(self.fake_B)
            self.real_buffer_2.append(self.real_B)

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

    def backwark_G(self):
        self.loss_G_mse = self.criterion_mse(self.fake_B, self.real_B)

        lambda_texture = self.opt.lambda_texture
        if lambda_texture > 0:

            if self.opt.texture_criterion == 'attention':
                self.loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor, self.attention)
            elif self.opt.texture_criterion == 'max':
                loss_texture = _texture_loss(self.fake_B, self.real_B, self.opt, self.grid_extractor)  # , self.texture_extractor

                # compute the loss function by averaging over the batch
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

        # Total loss
        self.loss_G = (
                self.loss_G_mse
                + self.loss_texture
                + self.loss_perceptual
        )

        self.optimizer_G.zero_grad()
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.backwark_G()
        self.optimizer_G.step()

    def update_learning_rate(self):
        # Update learning rates
        self.lr_scheduler_G.step()

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
            "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f] ETA: %s"
            % (epoch, self.opt.n_epochs, dataset_len, i, self.loss_G.item(), time_left)
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
        torch.save(self.netG.state_dict(), f"{self.save_dir}/E1_ep{epoch}_{self.opt.experiment_name}.pth")

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
                load_filename = 'net_%s_ep%s_unit_%s.pth' % (name, epoch, exp)
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

    def compute_FrechetInceptionDistance(self):
        real_buffer = torch.cat(self.real_test_buffer, dim=0)
        fake_buffer = torch.cat(self.fake_test_buffer, dim=0)

        fid_score = self.fid_object.compute_fid(fake_buffer, real_buffer, self.opt.dataset_len)
        self.metrics_eval['FID'].append(fid_score)

    def track_metrics(self):
        for name in self.metric_names:
            if isinstance(name, str) and (name != 'FID_ImNet' or name != 'FID_random'):
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
                if isinstance(name, str) and name != 'FID_ImNet' and name != 'FID_random':
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
        np.save(f"{self.loss_dir}/attention_B.npy", np.array(self.attention_B))

    def save_attention_weights(self):
        np.save(f"{self.loss_dir}/weights.npy", np.array(self.weight))

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

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
        parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
        parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--hr_height", type=int, default=128, help="size of image height")
        parser.add_argument("--hr_width", type=int, default=128, help="size of image width")
        parser.add_argument("--channels", type=int, default=3, help="number of image channels")
        parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
        parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
        parser.add_argument("--lambda_GAN", type=int, default=0.001, help="GAN loss")
        parser.add_argument("--lambda_texture", type=int, default=0.001, help="Texture loss weight")
        parser.add_argument("--lambda_perceptual", type=int, default=0.1, help="Perceptual loss weight")
        parser.add_argument("--lambda_texture", type=int, default=0.001, help="Texture loss weight")
        parser.add_argument("--lambda_perceptual", type=int, default=0.1, help="Perceptual loss weight")
        parser.add_argument('--perceptual_layers', type=str, default='all', help='choose the perceptual layers.')
        parser.add_argument('--vgg_pretrained', type=str, default=True, help='pretraining flag.')
        parser.add_argument("--texture_criterion", type=str, default="max", help="select the aggregation rule")
        parser.add_argument('--texture_offsets', type=str, default="all", help='texture offsets.')
        parser.add_argument('--image_folder', type=str, default=None, help='folder to save images during training')
        parser.add_argument('--metric_folder', type=str, default=None, help='folder to save metrics')
        parser.add_argument('--loss_folder', type=str, default=None, help='folder to save losses')
        parser.add_argument('--test_folder', type=str, default=None, help='folder to save test images')
        parser.add_argument('--test', type=str, default="test_1", help='folder to save test images')
        parser.add_argument('--experiment_name', type=str, default=None, help='experiment name')
        self.isTrain = True
        return parser
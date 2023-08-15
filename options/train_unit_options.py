from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        parser.add_argument("--dataset_name", type=str, default="apple2orange", help="name of the dataset")
        parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_height", type=int, default=256, help="size of image height")
        parser.add_argument("--img_width", type=int, default=256, help="size of image width")
        parser.add_argument("--channels", type=int, default=3, help="number of image channels")
        parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator samples")
        parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
        parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
        parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
        parser.add_argument("--lambda_0", type=float, default=10, help="GAN loss weight")
        parser.add_argument("--lambda_1", type=int, default=0.1, help="KL (encoded images) weight")
        parser.add_argument("--lambda_2", type=int, default=100, help="ID pixel-wise weight")
        parser.add_argument("--lambda_3", type=int, default=0.1, help="KL (encoded translated images) weight")
        parser.add_argument("--lambda_4", type=int, default=100, help="Cycle pixel-wise weight")


        self.isTrain = True
        return parser

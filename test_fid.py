"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer_offline import Visualizer
from util.util import save_ordered_dict_as_csv
from tqdm import tqdm
import torch
from metrics.FID import *

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots

    fake_buffer = torch.load(f'/Volumes/Untitled/{opt.test_folder}/fake_buffer_{opt.test}_epoch{opt.epoch}.pth', map_location=torch.device('cpu'))
    real_buffer = torch.load(f'/Volumes/Untitled/{opt.test_folder}/real_buffer_{opt.test}_epoch{opt.epoch}.pth', map_location=torch.device('cpu'))

    print(f"len: {len(fake_buffer)}")
    metric_obj = GANMetrics('cpu', detector_name='inceptionv3', batch_size=64)
    fid = metric_obj.compute_fid(fake_buffer, real_buffer, len(fake_buffer))
    fid = {"fid": fid}
    save_ordered_dict_as_csv(fid, f"/Volumes/Untitled/{opt.test_folder}/fid_{opt.test}.csv")

    # python3 ./test_fid.py --name pix-2-pix_baseline --dataroot False --gpu_ids -1 --no_html --display_id 0 --test_folder test_pix2pix_texture_avg_window_5 --epoch 50 --test test_1
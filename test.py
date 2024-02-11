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
import random

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
from metrics.mse_psnr_ssim_vif import azimuthalAverage
from util.util import save_ordered_dict_as_csv
from tqdm import tqdm
import numpy as np
import json

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    opt.isTrain = False
    opt.epoch = 50
    opt.serial_batches = True
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup2(opt, "/Volumes/Untitled/models_cycleGAN/")  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    model.eval()

    ####################################
    # Test_2 (8 patients)
    opt.text_file = "./data/mayo_test_ext.csv"  # load the csv file containing test data info
    opt.dataset_mode = "mayo"
    test_2 = create_dataset(opt)
    print(len(test_2))
    ####################################
    # Test_mayo_ext_2 (5 patients)
    opt.text_file = "./data/mayo_test_ext_2.csv"  # load the csv file containing test data info
    opt.dataset_mode = "mayo"
    test_mayo_ext_2 = create_dataset(opt)
    print(len(test_mayo_ext_2))
    ####################################
    # elcap_complete (50 patients)
    opt.text_file = "./data/elcap_data.csv"  # load the csv file containing test data info
    opt.dataset_mode = "LIDC_IDRI"
    elcap_complete = create_dataset(opt)
    print(len(elcap_complete))
    ####################################
    # Test 3 (8 patient from LIDC/IDRI)
    opt.text_file = "./data/LIDC_test.csv"  # load the csv file containing test data info
    opt.dataset_mode = "LIDC_IDRI"
    dataset_test_3 = create_dataset(opt)
    print(len(dataset_test_3))
    ####################################
    # Test 3 (8 patient from LIDC/IDRI)
    opt.text_file = "./data/haralick_magnitude.csv"  # load the csv file containing test data info
    opt.dataset_mode = "mayo"
    dataset_test_4 = create_dataset(opt)
    print(len(dataset_test_3))


    opt.dataset_len = len(dataset_test_4)
    opt.test = 'test_2'
    opt.dataset_mode = "mayo"
    print(f"Test: {opt.test}")
    random.seed(42)
    list_index = [102]   # random.sample(range(2996), 20)
    for j, data_test in tqdm(enumerate(dataset_test_4)):
            model.set_input(data_test)  # unpack data from data loader
            model.test()  # run inference"""

    model.save_haralicks()
    # model.save_template(opt.epoch)
    # model.fid_compute()
    # model.save_metrics_per_patient(opt.epoch)
    """opt.dataset_len = len(test_2)
    opt.test = 'test_2'
    opt.dataset_mode = "mayo"
    print(f"Test: {opt.test}")
    random.seed(42)
    list_index = [102]   # random.sample(range(2996), 20)
    for j, data_test in tqdm(enumerate(test_2)):
            model.set_input(data_test)  # unpack data from data loader
            model.test2(j, list_index)  # run inference"""

    # model.save_template(opt.epoch)
    # model.fid_compute()
    # model.save_metrics_per_patient(opt.epoch)

    ####################################
    """opt.dataset_len = len(dataset_test_3)
    opt.test = 'test_3'
    opt.dataset_mode = "LIDC_IDRI"
    random.seed(42)
    list_index = [1516]  # random.sample(range(1831), 20)
    print(f"Test: {opt.test}")
    for j, data_test in tqdm(enumerate(dataset_test_3)):
            model.set_input(data_test)  # unpack data from data loader
            model.test2(j, list_index)  # run inference"""

    # model.save_template(opt.epoch)
    # model.fid_compute()
    # model.save_metrics_per_patient(opt.epoch)"""
    ####################################
    """opt.dataset_len = len(elcap_complete)
    opt.test = 'elcap_complete'
    opt.dataset_mode = "LIDC_IDRI"
    random.seed(42)
    list_index = [11087]  # random.sample(range(12360), 20)
    print(f"Test: {opt.test}")
    for j, data_test in tqdm(enumerate(elcap_complete)):
            model.set_input(data_test)  # unpack data from data loader
            model.test2(j, list_index)  # run inference

    # model.save_template(opt.epoch)
    # model.fid_compute()
    # model.save_metrics_per_patient(opt.epoch)"""
    """opt.dataset_len = len(elcap_complete)
    opt.dataset_mode = "LIDC_IDRI"
    opt.test = 'elcap_complete'
    print(f"Test: {opt.test}")
    for j, data_test in tqdm(enumerate(elcap_complete)):
            if j <= 4:
                model.set_input(data_test)  # unpack data from data loader
                model.test2()  # run inference

    # model.save_image_buffers(opt.epoch)
    model.save_noise_metrics(opt.epoch)
    ####################################
    opt.dataset_len = len(dataset_test_3)
    opt.dataset_mode = "LIDC_IDRI"
    opt.test = 'test_3'
    print(f"Test: {opt.test}")
    for j, data_test in tqdm(enumerate(dataset_test_3)):
            model.set_input(data_test)  # unpack data from data loader
            model.test(j)  # run inference"""

    # model.save_image_buffers(opt.epoch)
    # model.save_noise_metrics(opt.epoch)
    # model.fid_compute(opt.epoch)
    # model.save_raps_per_patient(opt.epoch)
    # model.save_metrics_per_patient(opt.epoch)





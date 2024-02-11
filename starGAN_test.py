import datetime
import time
from data import create_dataset
import torch
from options.starGAN_options import TrainOptions
from models.starGAN_model import StarGANModel
from tqdm import tqdm
cuda = True if torch.cuda.is_available() else False

if __name__ == '__main__':

    # ------------------------------
    # Train dataset
    opt = TrainOptions().parse()  # get training options
    opt.epoch = 0
    opt.text_file = "./data/mayo_training_9pat.csv"
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (train dataset)
    dataset_size = len(dataset)  # get the number of images in the training dataset.
    print('The number of training images = %d' % dataset_size)
    # ------------------------------
    # Test 1 (1 patient from Mayo dataset)
    opt.text_file = "./data/mayo_test_1p.csv"  # load the csv file containing test data info
    opt.isTrain = False
    opt.test = "test_1"
    opt.serial_batches = True
    opt.batch_size = 1
    dataset_test = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (test dataset)
    print(len(dataset_test))
    # ------------------------------
    # Test 2 (8 patient from Mayo dataset extended)
    opt.text_file = "./data/mayo_test_ext.csv"  # load the csv file containing test data info
    opt.test = "test_2"
    dataset_test_2 = create_dataset(opt)
    print(len(dataset_test_2))
    # ------------------------------
    # elcap_complete (ELCAP dataset 50 patients)
    opt.text_file = "./data/ELCAP.csv"  # load the csv file containing test data info
    opt.test = "elcap_complete"
    elcap_complete = create_dataset(opt)
    print(len(elcap_complete))
    # ------------------------------
    # Test 3 (8 patient from LIDC/IDRI)
    opt.text_file = "./data/LIDC_test.csv"  # load the csv file containing test data info
    opt.test = "test_3"
    dataset_test_3 = create_dataset(opt)
    print(len(dataset_test_3))
    # ------------------------------
    epoch = 50
    exp = "ep_50_baseline_1"
    model = StarGANModel(opt)
    model.load_networks_2(epoch, exp, "/Volumes/Untitled/models_starGAN")


    # ------
    #  Test
    # ------
    test_time = 0
    test_start = time.time()

    opt.serial_batches = True
    opt.batch_size = 1
    test_start = time.time()
    opt.isTrain = False
    model.eval()
    # ---------------------------------
    opt.dataset_len = len(dataset_test)
    opt.test = 'test_1'
    print(f"Test: {opt.test}")
    for j, data_test in enumerate(dataset_test):
        model.set_input((data_test, j))  # unpack data from data loader
        model.test()  # run inference

    model.save_metrics_per_patient(epoch)
    # ---------------------------------
    opt.dataset_len = len(dataset_test_2)
    opt.test = 'test_2'
    print(f"Test: {opt.test}")
    for j, data_test in enumerate(dataset_test_2):
        model.set_input((data_test, j))  # unpack data from data loader
        model.test()  # run inference

    model.fid_compute()
    model.save_metrics_per_patient(epoch)
    # ---------------------------------
    opt.dataset_len = len(elcap_complete)
    opt.test = 'elcap_complete'
    opt.dataset_mode = "LIDC_IDRI"
    print(f"Test: {opt.test}")
    for j, data_test in enumerate(elcap_complete):
        model.set_input((data_test, j))  # unpack data from data loader
        model.test()  # run inference

    model.save_raps_per_patient(epoch)
    model.save_metrics_per_patient(epoch)
    # ---------------------------------
    opt.dataset_len = len(dataset_test_3)
    opt.dataset_mode = "LIDC_IDRI"
    opt.test = 'test_3'
    print(f"Test: {opt.test}")
    for j, data_test in enumerate(dataset_test_3):
        model.set_input((data_test, j))  # unpack data from data loader
        model.test()  # run inference

    model.save_raps_per_patient(epoch)
    model.save_metrics_per_patient(epoch)



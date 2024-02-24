import datetime
import time
from data import create_dataset
import torch
from options.train_unit_options import TrainOptions
from models.unitmodified_model import UNITMODIFIEDModel
from loss_functions.early_stopping import EarlyStopping

cuda = True if torch.cuda.is_available() else False

if __name__ == '__main__':

    # ------------------------------
    # Train dataset
    opt = TrainOptions().parse()  # get training options
    opt.source = 'B30'
    opt.target = 'D45'
    opt.epoch = 1
    opt.text_file = "./data/mayo_1mmB30D45.csv"
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (train dataset)
    dataset_size = len(dataset)  # get the number of images in the training dataset.
    print('The number of training images = %d' % dataset_size)
    # ------------------------------
    # Test 1 (1 patient from Mayo dataset)
    opt.text_file = "./data/mayo_test_1mmB30D45.csv"  # load the csv file containing test data info
    opt.isTrain = False
    opt.serial_batches = True
    opt.dataset_mode = "LIDC_IDRI"
    opt.batch_size = 1
    dataset_test = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (test dataset)
    print(len(dataset_test))
    # ------------------------------
    model = UNITMODIFIEDModel(opt)

    # ----------
    #  Training
    # ----------
    opt.batch_size = 4
    opt.serial_batches = False
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        epoch_start_time = time.time()
        for i, batch in enumerate(dataset):
            opt.dataset_mode = "mayo"
            # Set model input
            model.set_input(batch)
            model.optimize_parameters()

            opt.serial_batches = True
            opt.batch_size = 1
            opt.dataset_mode = "LIDC_IDRI"
            test_start = time.time()
            opt.isTrain = False
            model.eval()
            # ---------------------------------
            opt.dataset_len = len(dataset_test)
            opt.test = 'test_1'
            print(f"Test: {opt.test}")
            model.average_adaiN()
            for j, data_test in enumerate(dataset_test):
                model.set_input(data_test)  # unpack data from data loader
                model.test()  # run inference

            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = epoch * len(dataset) + i
            batches_left = opt.n_epochs * len(dataset) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            model.print_current_loss(epoch, dataset_size, i, time_left)
        model.plot_current_losses(epoch, model.track_current_losses(), 'Loss_functions')

        #########################################
        """current_texture_loss = (model.get_current_losses()['cycle_texture_X1'] + model.get_current_losses()['cycle_texture_X2'])/2
        early_stopping_bool = early_stopping.should_save_checkpoint(current_texture_loss)    
        if early_stopping_bool:
            print(f"Early stopping triggered at epoch {epoch}.")
            model.save_networks("early_stopping")
            if opt.texture_criterion == 'attention':
                model.save_attention_maps()
                model.save_attention_weights()"""
        #########################################

        if epoch == 20 or epoch == 40 or epoch == 100 or epoch == 200:
            model.save_networks(epoch)

        # ------
        #  Test
        # ------
        test_time = 0
        test_start = time.time()
        if epoch == 1 or epoch == 5 or epoch == 10 or epoch == 40:
            opt.serial_batches = True
            opt.batch_size = 1
            opt.dataset_mode = "LIDC_IDRI"
            test_start = time.time()
            opt.isTrain = False
            model.eval()
            # ---------------------------------
            opt.dataset_len = len(dataset_test)
            opt.test = 'test_1'
            print(f"Test: {opt.test}")
            for j, data_test in enumerate(dataset_test):
                model.set_input(data_test)  # unpack data from data loader
                model.test()  # run inference

            model.save_test_images(epoch)
            # ---------------------------------
            opt.batch_size = 4
            opt.serial_batches = False
            model.train()

        test_end = time.time()
        test_time = test_end - test_start
        model.update_learning_rate()
        model.print_time(epoch, (time.time() - epoch_start_time) - test_time)

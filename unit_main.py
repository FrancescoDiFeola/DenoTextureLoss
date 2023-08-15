import os
import itertools
import datetime
import time
import sys
from models.unit_networks import *
from data import create_dataset
import torch
from options.train_unit_options import TrainOptions
from models.unit_model import UNITModel

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

    model = UNITModel(opt)

    # ----------
    #  Training
    # ----------
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataset):
            # Set model input
            model.set_input(batch)
            model.optimize_parameters()

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

        model.update_learning_rate()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            model.save_networks(epoch)

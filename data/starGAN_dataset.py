import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import pandas as pd
import pydicom
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


class StarGANDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.

    Here A stands for 'low-dose' and B stands for 'high-dose'
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """
        ####
        parser.add_argument('--text_file', type=str, default=None,
                            help='Path of the csv_file containing the directories of the images')
        parser.add_argument('--window_width', type=int, default=1400,
                            help='Window width specifies the rage of HU values to display')
        parser.add_argument('--window_center', type=int, default=-500,
                            help='It specifies the center of the selected HU window')
        parser.add_argument('--plot_verbose', help="Plot images.", type=bool, default=False)

        return parser

    @staticmethod
    def convert_in_hu(dicom_file):  # apply the linear transformations to get the HU values (y = m*x + q)
        image = dicom_file.pixel_array
        intercept = dicom_file.RescaleIntercept
        slope = dicom_file.RescaleSlope
        image = slope * image + intercept

        return image

    @staticmethod
    def normalize_img(x, lower=None, upper=None, data_range='-11'):
        if lower is None:
            lower = np.min(x)

        if upper is None:
            upper = np.max(x)

        x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))  # map between 0 and 1

        if data_range == '01':

            return x_norm
        else:
            return (2 * x_norm) - 1

    @staticmethod
    def plot_img(x, pname):
        x = x.detach().cpu().numpy()
        x = x[0, :, :]
        plt.imshow(x, cmap='gray')
        plt.title(pname)
        plt.axis('off')
        plt.show()

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        annotations = self.opt.text_file
        if opt.isTrain or ((not self.opt.isTrain) and (self.opt.test == "test_3" or self.opt.test == "elcap_complete")):
            self.annotations_df = pd.read_csv(annotations)
            self.dataset_len = len(self.annotations_df)  # get the size of dataset A
        else:
            annotations_df = pd.read_csv(annotations)
            self.annotations_A = annotations_df.loc[annotations_df['domain'] == self.opt.source].reset_index(drop=True)
            self.annotations_B = annotations_df.loc[annotations_df['domain'] == self.opt.target].reset_index(drop=True)
            self.dataset_len = len(self.annotations_A)  # get the size of dataset A

        self.window_width = opt.window_width
        self.window_center = opt.window_center
        self.plot_verbose = opt.plot_verbose

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.opt.isTrain or ((not self.opt.isTrain) and (self.opt.test == "test_3" or self.opt.test == "elcap_complete")):
            index = index % self.dataset_len
            img_path = self.annotations_df['path_slice'].iloc[index]  # make sure index is within the range
            img = pydicom.dcmread(img_path)

            # apply image transformation
            img = self.transforms(img)
            id = self.annotations_df['patient'].iloc[index]

            if self.annotations_df['domain'].iloc[index] == self.opt.source:
                label = torch.Tensor([0])
            elif self.annotations_df['domain'].iloc[index] == self.opt.target:
                label = torch.Tensor([1])
            else:
                raise "NotImplemented"

            return {'img': img, "label": label, "patient": id}

        elif (not self.opt.isTrain) and (self.opt.test == "test_1" or self.opt.test == "test_2"):
            index = index % self.dataset_len
            A_path = self.annotations_A['path_slice'].iloc[index]  # make sure index is within the range
            B_path = self.annotations_B['path_slice'].iloc[index]
            id = self.annotations_A['patient'].iloc[index]
            A_img = pydicom.dcmread(A_path)
            B_img = pydicom.dcmread(B_path)
            # apply image transformation
            A = self.transforms(A_img)
            B = self.transforms(B_img)
            label_A = torch.Tensor([0])
            label_B = torch.Tensor([1])

            return {'img_A': A, 'img_B': B, 'label_A': label_A, 'label_B': label_B, "patient": id}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.dataset_len

    def window_image(self, hu_img):  # image windowing
        img_w = hu_img.copy()
        img_min = self.window_center - self.window_width // 2
        img_max = self.window_center + self.window_width // 2
        img_w[img_w < img_min] = img_min
        img_w[img_w > img_max] = img_max

        return img_w

    def transforms(self, dicom, tensor_output=True):
        x = self.convert_in_hu(dicom)

        if self.opt.windowing == False:
            x = np.clip(x, -1024, 3071)
        else:
            x = self.window_image(x)

        x = self.normalize_img(x)
        x = cv2.resize(x, (self.opt.load_size, self.opt.load_size))
        if tensor_output:
            x = torch.from_numpy(x)
            x = x.unsqueeze(dim=0)
            return x.float()
        else:
            return x.astype('float32')

from skimage.feature import graycomatrix, graycoprops
import cv2
import numpy as np
import math
from options.train_options import TrainOptions
from data import create_dataset
import torch
from data.storage import save_to_json
from tqdm import tqdm
torch.manual_seed(42)

if __name__ == "__main__":

    haralicks_A = {"contrast_1-0": [],
                 "contrast_1-45": [],
                 "contrast_1-90": [],
                 "contrast_1-135": [],
                 "dissimilarity_1-0": [],
                 "dissimilarity_1-45": [],
                 "dissimilarity_1-90": [],
                 "dissimilarity_1-135": [],
                 "homogeneity_1-0": [],
                 "homogeneity_1-45": [],
                 "homogeneity_1-90": [],
                 "homogeneity_1-135": [],
                 "ASM_1-0": [],
                 "ASM_1-45": [],
                 "ASM_1-90": [],
                 "ASM_1-135": [],
                 "energy_1-0": [],
                 "energy_1-45": [],
                 "energy_1-90": [],
                 "energy_1-135": [],
                 "correlation_1-0": [],
                 "correlation_1-45": [],
                 "correlation_1-90": [],
                 "correlation_1-135": [],
                 "contrast_3-0": [],
                 "contrast_3-45": [],
                 "contrast_3-90": [],
                 "contrast_3-135": [],
                 "dissimilarity_3-0": [],
                 "dissimilarity_3-45": [],
                 "dissimilarity_3-90": [],
                 "dissimilarity_3-135": [],
                 "homogeneity_3-0": [],
                 "homogeneity_3-45": [],
                 "homogeneity_3-90": [],
                 "homogeneity_3-135": [],
                 "ASM_3-0": [],
                 "ASM_3-45": [],
                 "ASM_3-90": [],
                 "ASM_3-135": [],
                 "energy_3-0": [],
                 "energy_3-45": [],
                 "energy_3-90": [],
                 "energy_3-135": [],
                 "correlation_3-0": [],
                 "correlation_3-45": [],
                 "correlation_3-90": [],
                 "correlation_3-135": [],
                 "contrast_5-0": [],
                 "contrast_5-45": [],
                 "contrast_5-90": [],
                 "contrast_5-135": [],
                 "dissimilarity_5-0": [],
                 "dissimilarity_5-45": [],
                 "dissimilarity_5-90": [],
                 "dissimilarity_5-135": [],
                 "homogeneity_5-0": [],
                 "homogeneity_5-45": [],
                 "homogeneity_5-90": [],
                 "homogeneity_5-135": [],
                 "ASM_5-0": [],
                 "ASM_5-45": [],
                 "ASM_5-90": [],
                 "ASM_5-135": [],
                 "energy_5-0": [],
                 "energy_5-45": [],
                 "energy_5-90": [],
                 "energy_5-135": [],
                 "correlation_5-0": [],
                 "correlation_5-45": [],
                 "correlation_5-90": [],
                 "correlation_5-135": [],
                 "contrast_7-0": [],
                 "contrast_7-45": [],
                 "contrast_7-90": [],
                 "contrast_7-135": [],
                 "dissimilarity_7-0": [],
                 "dissimilarity_7-45": [],
                 "dissimilarity_7-90": [],
                 "dissimilarity_7-135": [],
                 "homogeneity_7-0": [],
                 "homogeneity_7-45": [],
                 "homogeneity_7-90": [],
                 "homogeneity_7-135": [],
                 "ASM_7-0": [],
                 "ASM_7-45": [],
                 "ASM_7-90": [],
                 "ASM_7-135": [],
                 "energy_7-0": [],
                 "energy_7-45": [],
                 "energy_7-90": [],
                 "energy_7-135": [],
                 "correlation_7-0": [],
                 "correlation_7-45": [],
                 "correlation_7-90": [],
                 "correlation_7-135": [],
                 }
    haralicks_B = {"contrast_1-0": [],
                   "contrast_1-45": [],
                   "contrast_1-90": [],
                   "contrast_1-135": [],
                   "dissimilarity_1-0": [],
                   "dissimilarity_1-45": [],
                   "dissimilarity_1-90": [],
                   "dissimilarity_1-135": [],
                   "homogeneity_1-0": [],
                   "homogeneity_1-45": [],
                   "homogeneity_1-90": [],
                   "homogeneity_1-135": [],
                   "ASM_1-0": [],
                   "ASM_1-45": [],
                   "ASM_1-90": [],
                   "ASM_1-135": [],
                   "energy_1-0": [],
                   "energy_1-45": [],
                   "energy_1-90": [],
                   "energy_1-135": [],
                   "correlation_1-0": [],
                   "correlation_1-45": [],
                   "correlation_1-90": [],
                   "correlation_1-135": [],
                   "contrast_3-0": [],
                   "contrast_3-45": [],
                   "contrast_3-90": [],
                   "contrast_3-135": [],
                   "dissimilarity_3-0": [],
                   "dissimilarity_3-45": [],
                   "dissimilarity_3-90": [],
                   "dissimilarity_3-135": [],
                   "homogeneity_3-0": [],
                   "homogeneity_3-45": [],
                   "homogeneity_3-90": [],
                   "homogeneity_3-135": [],
                   "ASM_3-0": [],
                   "ASM_3-45": [],
                   "ASM_3-90": [],
                   "ASM_3-135": [],
                   "energy_3-0": [],
                   "energy_3-45": [],
                   "energy_3-90": [],
                   "energy_3-135": [],
                   "correlation_3-0": [],
                   "correlation_3-45": [],
                   "correlation_3-90": [],
                   "correlation_3-135": [],
                   "contrast_5-0": [],
                   "contrast_5-45": [],
                   "contrast_5-90": [],
                   "contrast_5-135": [],
                   "dissimilarity_5-0": [],
                   "dissimilarity_5-45": [],
                   "dissimilarity_5-90": [],
                   "dissimilarity_5-135": [],
                   "homogeneity_5-0": [],
                   "homogeneity_5-45": [],
                   "homogeneity_5-90": [],
                   "homogeneity_5-135": [],
                   "ASM_5-0": [],
                   "ASM_5-45": [],
                   "ASM_5-90": [],
                   "ASM_5-135": [],
                   "energy_5-0": [],
                   "energy_5-45": [],
                   "energy_5-90": [],
                   "energy_5-135": [],
                   "correlation_5-0": [],
                   "correlation_5-45": [],
                   "correlation_5-90": [],
                   "correlation_5-135": [],
                   "contrast_7-0": [],
                   "contrast_7-45": [],
                   "contrast_7-90": [],
                   "contrast_7-135": [],
                   "dissimilarity_7-0": [],
                   "dissimilarity_7-45": [],
                   "dissimilarity_7-90": [],
                   "dissimilarity_7-135": [],
                   "homogeneity_7-0": [],
                   "homogeneity_7-45": [],
                   "homogeneity_7-90": [],
                   "homogeneity_7-135": [],
                   "ASM_7-0": [],
                   "ASM_7-45": [],
                   "ASM_7-90": [],
                   "ASM_7-135": [],
                   "energy_7-0": [],
                   "energy_7-45": [],
                   "energy_7-90": [],
                   "energy_7-135": [],
                   "correlation_7-0": [],
                   "correlation_7-45": [],
                   "correlation_7-90": [],
                   "correlation_7-135": [],
                   }

    def haralicks_extractor(x, dictionary):
        x = cv2.normalize(x.detach().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

        spatial_offset = [1, 3, 5, 7]
        angular_offset = [0, 45, 90, 135]

        for i in range(0, x.shape[0]):
            for idx_d, d in enumerate(spatial_offset):
                for idx_theta, theta in enumerate(angular_offset):
                    dictionary[f"contrast_{d}-{theta}"].append(
                        graycoprops(graycomatrix(x[i, 0, :, :], distances=[d], angles=[theta * (math.pi / 180)], levels=256, symmetric=True, normed=True), "contrast")[0][0])
                    dictionary[f"dissimilarity_{d}-{theta}"].append(
                        graycoprops(graycomatrix(x[i, 0, :, :], distances=[d], angles=[theta * (math.pi / 180)], levels=256, symmetric=True, normed=True), "dissimilarity")[0][0])
                    dictionary[f"homogeneity_{d}-{theta}"].append(
                        graycoprops(graycomatrix(x[i, 0, :, :], distances=[d], angles=[theta * (math.pi / 180)], levels=256, symmetric=True, normed=True), "homogeneity")[0][0])
                    dictionary[f"ASM_{d}-{theta}"].append(
                        graycoprops(graycomatrix(x[i, 0, :, :], distances=[d], angles=[theta * (math.pi / 180)], levels=256, symmetric=True, normed=True), "ASM")[0][0])
                    dictionary[f"energy_{d}-{theta}"].append(
                        graycoprops(graycomatrix(x[i, 0, :, :], distances=[d], angles=[theta * (math.pi / 180)], levels=256, symmetric=True, normed=True), "energy")[0][0])
                    dictionary[f"correlation_{d}-{theta}"].append(
                        graycoprops(graycomatrix(x[i, 0, :, :], distances=[d], angles=[theta * (math.pi / 180)], levels=256, symmetric=True, normed=True), "correlation")[0][0])

    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (train dataset)
    dataset_size = len(dataset)  # get the number of images in the training dataset.
    print('The number of training images = %d' % dataset_size)

    for i, data in tqdm(enumerate(dataset)):

        haralicks_extractor(data['A'], haralicks_A)
        haralicks_extractor(data['B'], haralicks_B)

    save_to_json(haralicks_A, "which_haralick_A")
    save_to_json(haralicks_B, "which_haralick_B")
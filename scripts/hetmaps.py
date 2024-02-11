import os
import torch
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec
import cv2
from tqdm import tqdm
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis
from scipy.stats import skew


def difference_image_grid(image_list1, image_list2, title):
    # Calculate grid dimensions
    num_images = len(image_list1)
    num_cols = int(num_images ** 0.5)
    num_rows = (num_images + num_cols - 1) // num_cols

    # Create a grid of subplots without spaces
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0, hspace=0)

    for i, img1 in enumerate(zip(image_list1, image_list2)):
        img = abs(img1[0][0, 0, :, :] - img1[1][0, 0, :, :])
        ax = plt.subplot(gs[i])
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.title(f"{title}")
    plt.tight_layout()
    plt.show()


def image_grid(image_list, title):
    # Calculate grid dimensions
    num_images = len(image_list[560:570])
    print(num_images)
    num_cols = int(num_images ** 0.5)
    print(num_cols)
    num_rows = (num_images + num_cols - 1) // num_cols
    print(num_rows)

    # Create a grid of subplots without spaces
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(num_rows, num_cols, wspace=0, hspace=0)

    for i, img in enumerate(image_list[560:570]):
        ax = plt.subplot(gs[i])
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.title(f"{title}")
    plt.tight_layout()
    plt.show()


def intensity_histogram(image_list):
    # Convert images to NumPy arrays and rescale to [0, 1]
    image_arrays = [(image.numpy() + 1) / 2 for image in image_list]

    # Calculate histograms for each image
    histograms = [np.histogram(image.flatten(), bins=256, range=(0, 1))[0] for image in image_arrays]

    # Sum up histograms
    average_histogram = np.sum(histograms, axis=0)

    # Normalize to get the average histogram
    average_histogram = average_histogram.astype(float) / len(image_list)

    return average_histogram


def plot(average_histogram, average_histogram1, title):
    # Calculate correlation coefficient
    corr_coeff, _ = pearsonr(average_histogram, average_histogram1)
    print(f"Correlation Coefficient {title}:", corr_coeff)
    plt.figure(figsize=(10, 6))
    plt.plot(average_histogram, color='black')
    plt.plot(average_histogram1, color='red')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


def compute_image_gradient(image):
    # Convert the image to grayscale if it's in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate gradients in the x and y directions
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    return gradient_x, gradient_y, gradient_magnitude


if __name__ == "__main__":

    a = torch.rand(256, 256, 3)
    _, _, grad = compute_image_gradient(a.numpy())

    # HEATMAPS
    """"std_images = torch.load("/Users/francescodifeola/Desktop/standardized_test_1_epoch100.pth", map_location=torch.device('cpu'))
    not_std_images = torch.load("/Users/francescodifeola/Desktop/not_standardized_test_1_epoch100.pth", map_location=torch.device('cpu'))
    # plt.imshow(fake_images[0], cmap='gray')
    # plt.show()
    # plt.imshow(fake_images[560], cmap='gray')
    # plt.show()
    histogram1 = intensity_histogram(std_images[0:559])
    histogram2 = intensity_histogram(std_images[560:])

    histogram3 = intensity_histogram(not_std_images[0:559])
    histogram4 = intensity_histogram(not_std_images[560:])
    plot(histogram1, histogram2, "Average intensity histogram std")
    plot(histogram3, histogram4, "Average intensity histogram not std")

    # difference_image_grid(fake_images, real_images, "Difference heatmap")
    # image_grid(fake_images, "Fake images")
    # image_grid(fake_images, "Fake images")
    # image_grid(real_images, "Real images")


    for i in ['test_3', 'elcap_complete']:
        skweness_1 = torch.load(f"/Volumes/sandisk/UNIT/metrics/metrics_texture_avg_1/skweness_{i}_epoch50.pth", )
        skweness_2 = torch.load(f"/Volumes/sandisk/UNIT/metrics/metrics_texture_avg_2/skweness_{i}_epoch50.pth", )
        skweness_3 = torch.load(f"/Volumes/sandisk/UNIT/metrics/metrics_texture_avg_3/skweness_{i}_epoch50.pth", )
        avg = [sum(i)/len(i) for i in zip(skweness_1, skweness_2, skweness_3)]
        print(sum(avg)/len(avg))"""

    """tensors = []
    for idx, diff_img1 in tqdm(enumerate(zip(tensor_fake, tensor_real))):
           diff = abs(diff_img1[0][0, 0, :, :] - diff_img1[1][0, 0, :, :])

           tensors.append(diff)

    # Define the grid layout
    rows = 6
    columns = 5

    # Create a figure and axes for the grid
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(12, 15))
    plt.subplots_adjust(wspace=0, hspace=0)  # Remove any spacing

    for j in tensors:
        # print(kurtosis(j.flatten()))
        print(skew(j.flatten())[0:560])

    for i, tensor in enumerate(tensors):
        # Convert tensor to numpy array and reshape if needed
        image = tensor.numpy()
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # Convert from CHW to HWC format

        # Plot the image on the corresponding axis
        row = i // columns
        col = i % columns
        axes[row, col].imshow(image, cmap="gray")
        axes[row, col].axis('off')  # Turn off axis labels

    # Hide any remaining empty subplots
    for i in range(len(tensors), rows * columns):
        row = i // columns
        col = i % columns
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()"""

    # ATTENTION MAPS
    # attention1 = np.load("/Volumes/sandisk//loss_attention_weights/attention_X1.npy")
    # attention2 = np.load("/Volumes/sandisk/UNIT/loss_attention_weights_2/attention_X1.npy")
    # attention3 = np.load("/Volumes/sandisk/UNIT/loss_attention_weights_3/attention_X1.npy")
    # attention4 = np.load("/Volumes/sandisk/UNIT/loss_attention_weights/attention_X2.npy")
    # attention5 = np.load("/Volumes/sandisk/UNIT/loss_attention_weights_2/attention_X2.npy")
    # attention6 = np.load("/Volumes/sandisk/UNIT/loss_attention_weights_3/attention_X2.npy")
    # stacked_experiments = np.concatenate([attention1, attention2, attention3, attention4, attention5, attention6], axis=0)

    attention1 = np.load("/Volumes/sandisk/pix2pix_results/loss_pix2pix_texture_att_1/attention_B.npy")
    attention2 = np.load("/Volumes/sandisk/pix2pix_results/loss_pix2pix_texture_att2/attention_B.npy")
    attention3 = np.load("/Volumes/sandisk/pix2pix_results/loss_pix2pix_texture_att3/attention_B.npy")
    # print(attention1.shape, attention2.shape, attention3.shape)
    stacked_experiments = np.concatenate([attention1, attention2, attention3], axis=0)
    # print(stacked_tensors.shape)
    averaged_attention_map = np.mean(stacked_experiments, axis=(0, 1))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the grid with black shades
    cax = ax.matshow(averaged_attention_map, cmap='gray')

    spatial_offsets = [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7]
    angular_offsets = [0, 45, 90, 135, 0, 45, 90, 135, 0, 45, 90, 135, 0, 45, 90, 135]

    # Add labels for rows and columns
    row_labels = [f'{i}-{j}°' for i, j in zip(spatial_offsets, angular_offsets)]
    col_labels = [f'{i}-{j}°' for i, j in zip(spatial_offsets, angular_offsets)]

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate the x-axis labels for better visibility
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    plt.title("Pix2Pix attention map", fontsize=18)
    # Display the grid
    plt.show()

    """w1 = np.load("/Volumes/sandisk/pix2pix_results/loss_pix2pix_texture_att1/weight.npy")
    w2 = np.load("/Volumes/sandisk/pix2pix_results/loss_pix2pix_texture_att2/weight.npy")
    w3 = np.load("/Volumes/sandisk/pix2pix_results/loss_pix2pix_texture_att3/weight.npy")
    print(len(w1), len(w2), len(w3))
    avg_weight_pix2pix = [(i+j+t)/3 for i, j, t in zip(w1, w2, w3)]
    w4 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att13/weight_A.npy")
    w5 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att14/weight_A.npy")
    w6 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att15/weight_A.npy")
    w7 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att13/weight_B.npy")
    w8 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att14/weight_B.npy")
    w9 = np.load("/Volumes/sandisk/cycleGAN_results/loss_texture_att15/weight_B.npy")
    print(len(w4), len(w5), len(w6), len(w7), len(w8), len(w9))
    avg_weight_cycle = [(a+b+c+d+e+f)/6 for a, b, c, d, e, f in zip(w4, w5, w6, w7, w8, w9)]
    w10 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_1/weight_X1.npy")
    w11 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_2/weight_X1.npy")
    w12 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_3/weight_X1.npy")
    w13 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_1/weight_X2.npy")
    w14 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_2/weight_X2.npy")
    w15 = np.load("/Volumes/sandisk/UNIT/loss_texture_att_early_3/weight_X2.npy")
    print(len(w10), len(w10), len(w10), len(w10), len(w10), len(w10))
    avg_weight_unit = [(t1+t2+t3+t4+t5+t6)/6 for t1, t2, t3, t4, t5, t6 in zip(w10, w11, w12, w13, w14, w15)]
    plt.plot(avg_weight_cycle)
    plt.plot(avg_weight_pix2pix)
    plt.plot(avg_weight_unit)
    plt.legend(["cycleGAN", "Pix2Pix", "UNIT"])
    plt.title("Average attention loss weight")
    plt.grid()
    plt.show()"""

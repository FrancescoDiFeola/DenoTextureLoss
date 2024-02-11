import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import cv2
from data.storage import *
from scipy.signal import correlate2d
from torchmetrics.functional.image import image_gradients
import pywt
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis
from scipy.stats import skew


def create_template(dim, mean, std, template="gaussian"):
    np.random.seed(42)
    if template == "gaussian":
        template_ = np.random.normal(mean, std, (dim, dim)).astype(np.float32)
        template_ = np.clip(template_, -1.0, 1.0)
    else:
        raise NotImplemented
    return template_


def template_matching(img, template):
    for t in [96]:

        w, h = template.shape[::-1]
        # All the 6 methods for comparison in a list
        methods = ['cv.TM_CCORR_NORMED']  # 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',, 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'
        for meth in methods:
            method = eval(meth)
            # Apply template Matching
            res = cv.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            avg_match = np.mean(res)

    return min_val, max_val, avg_match


def _kernel_density_estimate(img1, img3):  # , img3
    data1 = img1  # .flatten()
    # data2 = img2  #.flatten()
    data3 = img3  # .flatten()

    # Create a KDE plot
    sns.kdeplot(data1, color='blue')  # , shade=True
    # sns.kdeplot(data2, shade=True, color='red')
    sns.histplot(data3, color='orange')
    # Customize the plot
    plt.legend(["deno", "HD"])
    plt.title("Kernel Density Estimate (KDE) Plot")
    plt.xlabel("Value")
    plt.ylabel("Density")

    # Display the plot
    plt.show()


def kernel_density_estimate(d):  # , img3
    sns.kdeplot(d["baseline"]["hd"], color='#33a02c')
    sns.kdeplot(d["baseline"]["ld"], color='#e41a1c')
    sns.kdeplot(d["baseline"]["deno"], color='#1f78b4')
    sns.kdeplot(d["perceptual"]["deno"], color='#ff7f00')
    sns.kdeplot(d["autoencoder"]["deno"], color='#6a3d9a')
    sns.kdeplot(d["ssim"]["deno"], color='#b15928')
    sns.kdeplot(d["edge"]["deno"], color='#ff00ff')
    sns.kdeplot(d["texture_max"]["deno"], color='#4daf4a')
    sns.kdeplot(d["texture_avg"]["deno"], color='#ffd700')
    sns.kdeplot(d["texture_Frob"]["deno"], color='gray')
    sns.kdeplot(d["texture_att"]["deno"], color='black')
    # sns.kdeplot(d["perceptual"]["deno"], color='#00ced1')  # Different color for the second appearance of 'perceptual'

    # plt.grid()

    # Create a KDE plot
    # sns.histplot(data1, color='blue')  # , shade=True
    # sns.kdeplot(data2, shade=True, color='red')
    # sns.histplot(data3, color='orange')
    # Customize the plot
    plt.legend(["HDCT", "LDCT", "baseline", "VGG-16", "AE-CT", "SSIM", "EDGE", "MSTLF-max", "MSTLF-average", "MSLTF-Frobenius", "MSTLF-attention"])
    # plt.title("Kernel Density Estimate (KDE) Plot (Test set 4, UNIT)")
    plt.ylim(0, 50)
    plt.xlim(0.75, 1.03)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig('/Users/francescodifeola/Desktop/cycleGAN_test_2_tm.pdf', format='pdf')
    # Display the plot
    plt.show()


def draw_grid(img, crop_size, margin):
    cv.rectangle(img[0, 0, :, :], (margin, margin), (margin + crop_size, margin + crop_size), 255, 2)
    cv.rectangle(img[0, 0, :, :], (margin + 2 * crop_size, margin), (margin + 3 * crop_size, margin + crop_size), 255, 2)
    cv.rectangle(img[0, 0, :, :], (margin + 4 * crop_size, margin), (margin + 5 * crop_size, margin + crop_size), 255, 2)

    cv.rectangle(img[0, 0, :, :], (margin, margin + 2 * crop_size), (margin + crop_size, margin + 3 * crop_size), 255, 2)
    cv.rectangle(img[0, 0, :, :], (margin + 2 * crop_size, margin + 2 * crop_size), (margin + 3 * crop_size, margin + 3 * crop_size), 255, 2)
    cv.rectangle(img[0, 0, :, :], (margin + 4 * crop_size, margin + 2 * crop_size), (margin + 5 * crop_size, margin + 3 * crop_size), 255, 2)

    cv.rectangle(img[0, 0, :, :], (margin, margin + 4 * crop_size), (margin + crop_size, margin + 5 * crop_size), 255, 2)
    cv.rectangle(img[0, 0, :, :], (margin + 2 * crop_size, margin + 4 * crop_size), (margin + 3 * crop_size, margin + 5 * crop_size), 255, 2)
    cv.rectangle(img[0, 0, :, :], (margin + 4 * crop_size, margin + 4 * crop_size), (margin + 5 * crop_size, margin + 5 * crop_size), 255, 2)
    plt.imshow(img[0, 0, :, :], cmap='gray', vmin=-1, vmax=1)
    plt.ylim(0, 70)
    plt.show()


# TEMPLATE MATCHING
def template_matching(ld, hd, deno, template, method='cv.TM_CCOEFF_NORMED'):
    method = eval(method)
    res_1 = cv.matchTemplate(ld[0, 0, :, :], template, method)
    min_val_1, max_val_1, min_loc_1, max_loc_1 = cv.minMaxLoc(res_1)
    res_2 = cv.matchTemplate(deno[0, 0, :, :], template, method)
    min_val_2, max_val_2, min_loc_2, max_loc_2 = cv.minMaxLoc(res_2)
    res_3 = cv.matchTemplate(hd[0, 0, :, :], template, method)
    min_val_3, max_val_3, min_loc_3, max_loc_3 = cv.minMaxLoc(res_3)

    return {"ld": [min_val_1, max_val_1, min_loc_1, max_loc_1], "deno": [min_val_2, max_val_2, min_loc_2, max_loc_2], "hd": [min_val_3, max_val_3, min_loc_3, max_loc_3]}


if __name__ == "__main__":

    # Plot template matching KDE
    data1 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_baseline_tm_1/tm_test_2_epoch50")
    data2 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_baseline_tm_2/tm_test_2_epoch50")
    data3 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_baseline_tm_3/tm_test_2_epoch50")
    data4 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_tm_4/tm_test_2_epoch50")
    data5 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_tm_5/tm_test_2_epoch50")
    data6 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_perceptual_tm_6/tm_test_2_epoch50")
    data7 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_max_tm_1/tm_test_2_epoch50")
    data8 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_max_tm_2/tm_test_2_epoch50")
    data9 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_max_tm_3/tm_test_2_epoch50")
    data10 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_avg_tm_1/tm_test_2_epoch50")
    data11 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_avg_tm_2/tm_test_2_epoch50")
    data12 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_avg_tm_3/tm_test_2_epoch50")
    data13 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_Frob_tm_1/tm_test_2_epoch50")
    data14 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_Frob_tm_2/tm_test_2_epoch50")
    data15 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_Frob_tm_3/tm_test_2_epoch50")
    data16 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_att_tm_1/tm_test_2_epoch50")
    data17 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_att_tm_1/tm_test_2_epoch50")
    data18 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_texture_att_tm_3/tm_test_2_epoch50")
    data19 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_autoencoder_1/tm_test_2_epoch50")
    data20 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_autoencoder_2/tm_test_2_epoch50")
    data21 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_autoencoder_3/tm_test_2_epoch50")
    data22 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_ssim_1/tm_test_2_epoch50")
    data23 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_ssim_2/tm_test_2_epoch50")
    data24 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_ssim_3/tm_test_2_epoch50")
    data25 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_edge_1/tm_test_2_epoch50")
    data26 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_edge_2/tm_test_2_epoch50")
    data27 = load_from_json(f"/Volumes/Untitled/results_per_patient/cycleGAN/metrics_edge_3/tm_test_2_epoch50")

    d_ = {"baseline": {"deno": [], "hd": [], "ld": []},
          "texture_max": {"deno": [], "hd": [], "ld": []},
          "texture_avg": {"deno": [], "hd": [], "ld": []},
          "texture_Frob": {"deno": [], "hd": [], "ld": []},
          "texture_att": {"deno": [], "hd": [], "ld": []},
          "perceptual": {"deno": [], "hd": [], "ld": []},
          "autoencoder": {"deno": [], "hd": [], "ld": []},
          "ssim": {"deno": [], "hd": [], "ld": []},
          "edge": {"deno": [], "hd": [], "ld": []},
          }

    ld = []
    hd = []
    deno = []
    for j in test_2_ids:
        for i in data1[j]["images"].keys():
            ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
            hd.append(data1[j]["images"][str(i)]['computed_values']['deno'][1])
            # deno.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
            data_avg = (data1[j]["images"][str(i)]['computed_values']['hd'][1] + data2[j]["images"][str(i)]['computed_values']['hd'][1] + data3[j]["images"][str(i)]['computed_values']['hd'][1])/3
            deno.append(data_avg)
            # hd.append(data2[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data2[j]["images"][str(i)]['computed_values']['deno'][1])
            # hd.append(data3[j]["images"][str(i)]["computed_values"]['hd'])
            # deno.append(data3[j]["images"][str(i)]["computed_values"]['deno'])
    # print(sum(ld)/len(ld), sum(deno)/len(deno), sum(hd)/len(hd))
    # print(sum(deno)/len(deno), sum(ld)/len(ld))
    d_["baseline"]["deno"] = deno
    d_["baseline"]["hd"] = hd
    d_["baseline"]["ld"] = ld

    ld = []
    hd = []
    deno = []
    for j in test_2_ids:
        for i in data1[j]["images"].keys():
            # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
            # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
            #deno.append(data4[j]["images"][str(i)]['computed_values']['hd'][1])
            data_avg = (data4[j]["images"][str(i)]['computed_values']['hd'][1] + data5[j]["images"][str(i)]['computed_values']['hd'][1] + data6[j]["images"][str(i)]['computed_values']['hd'][1])/3
            deno.append(data_avg)
            # hd.append(data5[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data5[j]["images"][str(i)]['computed_values']['deno'][1])
            # hd.append(data6[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data6[j]["images"][str(i)]['computed_values']['deno'][1])

    d_["perceptual"]["deno"] = deno
    d_["perceptual"]["hd"] = hd
    d_["perceptual"]["ld"] = ld

    ld = []
    hd = []
    deno = []
    for j in test_2_ids:
        for i in data1[j]["images"].keys():
            # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
            #hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data6[j]["images"][str(i)]['computed_values']['hd'][1])
            data_avg = (data7[j]["images"][str(i)]['computed_values']['hd'][1] + data8[j]["images"][str(i)]['computed_values']['hd'][1] + data9[j]["images"][str(i)]['computed_values']['hd'][1])/3
            deno.append(data_avg)
            # hd.append(data8[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data8[j]["images"][str(i)]['computed_values']['deno'][1])
            # hd.append(data9[j]["images"][str(i)]['computed_values']['hd'])
            # deno.append(data9[j]["images"][str(i)]['computed_values']['deno'])

    d_["texture_max"]["deno"] = deno
    d_["texture_max"]["hd"] = hd
    d_["texture_max"]["ld"] = ld

    ld = []
    hd = []
    deno = []
    for j in test_2_ids:
        for i in data1[j]["images"].keys():
            #ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
            # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data11[j]["images"][str(i)]['computed_values']['hd'][1])
            data_avg = (data10[j]["images"][str(i)]['computed_values']['hd'][1] + data11[j]["images"][str(i)]['computed_values']['hd'][1] + data12[j]["images"][str(i)]['computed_values']['hd'][1])/3
            deno.append(data_avg)
            # hd.append(data11[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data11[j]["images"][str(i)]['computed_values']['deno'][1])
            # hd.append(data12[j]["images"][str(i)]['computed_values']['hd'])
            # deno.append(data12[j]["images"][str(i)]['computed_values']['deno'])

    d_["texture_avg"]["deno"] = deno
    d_["texture_avg"]["hd"] = hd
    d_["texture_avg"]["ld"] = ld

    ld = []
    hd = []
    deno = []
    for j in test_2_ids:
        for i in data1[j]["images"].keys():
            # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
            # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
            #deno.append(data14[j]["images"][str(i)]['computed_values']['hd'][1])
            data_avg = (data13[j]["images"][str(i)]['computed_values']['hd'][1] + data14[j]["images"][str(i)]['computed_values']['hd'][1] + data15[j]["images"][str(i)]['computed_values']['hd'][1])/3
            deno.append(data_avg)
            # hd.append(data14[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data14[j]["images"][str(i)]['computed_values']['deno'][1])
            # hd.append(data15[j]["images"][str(i)]['computed_values']['hd'])
            # deno.append(data15[j]["images"][str(i)]['computed_values']['deno'])

    d_["texture_Frob"]["deno"] = deno
    d_["texture_Frob"]["hd"] = hd
    d_["texture_Frob"]["ld"] = ld

    ld = []
    hd = []
    deno = []
    for j in test_2_ids:
        for i in data1[j]["images"].keys():
            # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
            # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
            data_avg = (data16[j]["images"][str(i)]['computed_values']['hd'][1] + data17[j]["images"][str(i)]['computed_values']['hd'][1] + data18[j]["images"][str(i)]['computed_values']['hd'][1])/3
            deno.append(data_avg)
            # hd.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data17[j]["images"][str(i)]['computed_values']['deno'][1])
            # hd.append(data18[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data18[j]["images"][str(i)]['computed_values']['deno'][1])

    d_["texture_att"]["deno"] = deno
    d_["texture_att"]["hd"] = hd
    d_["texture_att"]["ld"] = ld

    ld = []
    hd = []
    deno = []
    for j in test_2_ids:
        for i in data1[j]["images"].keys():
            # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
            # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
            print(j)
            print(data19[j]["images"][str(i)]['computed_values']['deno'][1])
            print(data20[j]["images"][str(i)]['computed_values']['deno'][1])
            print(data21[j]["images"][str(i)]['computed_values']['deno'][1])
            data_avg = (data19[j]["images"][str(i)]['computed_values']['hd'][1] + data20[j]["images"][str(i)]['computed_values']['hd'][1] + data21[j]["images"][str(i)]['computed_values']['hd'][1])/3
            deno.append(data_avg)
            # hd.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data17[j]["images"][str(i)]['computed_values']['deno'][1])
            # hd.append(data18[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data18[j]["images"][str(i)]['computed_values']['deno'][1])

    d_["autoencoder"]["deno"] = deno
    d_["autoencoder"]["hd"] = hd
    d_["autoencoder"]["ld"] = ld

    ld = []
    hd = []
    deno = []
    for j in test_2_ids:
        for i in data1[j]["images"].keys():
            # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
            # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
            data_avg = (data22[j]["images"][str(i)]['computed_values']['hd'][1] + data23[j]["images"][str(i)]['computed_values']['hd'][1] + data24[j]["images"][str(i)]['computed_values']['hd'][1])/3
            deno.append(data_avg)
            # hd.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data17[j]["images"][str(i)]['computed_values']['deno'][1])
            # hd.append(data18[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data18[j]["images"][str(i)]['computed_values']['deno'][1])

    d_["ssim"]["deno"] = deno
    d_["ssim"]["hd"] = hd
    d_["ssim"]["ld"] = ld

    ld = []
    hd = []
    deno = []
    for j in test_2_ids:
        for i in data1[j]["images"].keys():
            # ld.append(data1[j]["images"][str(i)]['computed_values']['ld'][1])
            # hd.append(data1[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
            data_avg = (data25[j]["images"][str(i)]['computed_values']['hd'][1] + data26[j]["images"][str(i)]['computed_values']['hd'][1] + data27[j]["images"][str(i)]['computed_values']['hd'][1])/3
            deno.append(data_avg)
            # hd.append(data17[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data17[j]["images"][str(i)]['computed_values']['deno'][1])
            # hd.append(data18[j]["images"][str(i)]['computed_values']['hd'][1])
            # deno.append(data18[j]["images"][str(i)]['computed_values']['deno'][1])

    d_["edge"]["deno"] = deno
    d_["edge"]["hd"] = hd
    d_["edge"]["ld"] = ld

    kernel_density_estimate(d_)


    """margin = 48
    crop_size = 32
    # x_coord = [(margin, margin + crop_size), (margin + 2 * crop_size, margin + 3 * crop_size), (margin + 4 * crop_size, margin + 5 * crop_size),
    #            (margin, margin + crop_size), (margin + 2 * crop_size, margin + 3 * crop_size), (margin + 4 * crop_size, margin + 5 * crop_size),
    #            (margin, margin + crop_size), (margin + 2 * crop_size, margin + 3 * crop_size), (margin + 4 * crop_size, margin + 5 * crop_size)]

    # y_coord = [(margin, margin + crop_size), (margin, margin + crop_size), (margin, margin + crop_size),
    #            (margin + 2 * crop_size, margin + 3 * crop_size), (margin + 2 * crop_size, margin + 3 * crop_size), (margin + 2 * crop_size, margin + 3 * crop_size),
    #           (margin + 4 * crop_size, margin + 5 * crop_size), (margin + 4 * crop_size, margin + 5 * crop_size), (margin + 4 * crop_size, margin + 5 * crop_size)]

    # low_dose_img = np.array(Image.open("/Volumes/Untitled/denoised_CT/low-dose/C120/low_dose_pat_C120_119.jpg")).astype(np.uint8)
    low_dose_img = torch.load("/Users/francescodifeola/PycharmProjects/DenoTextureLoss/checkpoints/pix-2-pix_baseline/web/metrics_pix2pix_texture_att_diff_7/100low"
                              "-dose_test_2_epoch50.pth").numpy()
    # draw_grid(low_dose_img, crop_size, margin)
    # print(low_dose_img.shape, low_dose_img.dtype, np.min(low_dose_img), np.max(low_dose_img))
    # high_dose_img = np.array(Image.open("/Volumes/Untitled/denoised_CT/high-dose/C120/high_dose_pat_C120_119.jpg")).astype(np.uint8)
    high_dose_img = torch.load(
        "/Users/francescodifeola/PycharmProjects/DenoTextureLoss/checkpoints/pix-2-pix_baseline/web/metrics_pix2pix_texture_att_diff_7/100high-dose_test_2_epoch50.pth").numpy()
    # print(high_dose_img.shape, high_dose_img.dtype, np.min(high_dose_img), np.max(high_dose_img))

    # denoised_img = np.array(Image.open("/Volumes/Untitled/denoised_CT/denoised/C120/denoised_pat_C120_119.jpg")).astype(np.uint8)
    denoised_img = torch.load(
        "/Users/francescodifeola/PycharmProjects/DenoTextureLoss/checkpoints/pix-2-pix_baseline/web/metrics_pix2pix_texture_att_diff_7/100denoised_test_2_epoch50.pth").numpy()
    # print(denoised_img.shape, denoised_img.dtype, np.min(denoised_img), np.max(denoised_img))

    correlation1 = correlate2d(low_dose_img[0, 0, :, :], denoised_img[0, 0, :, :], mode='same', boundary='wrap')
    correlation2 = correlate2d(low_dose_img[0, 0, :, :], high_dose_img[0, 0, :, :], mode='same', boundary='wrap')
    correlation_coefficient1 = np.corrcoef(low_dose_img[0, 0, :, :].flatten(), denoised_img[0, 0, :, :].flatten())[0, 1]
    correlation_coefficient2 = np.corrcoef(low_dose_img[0, 0, :, :].flatten(), high_dose_img[0, 0, :, :].flatten())[0, 1]
    grad_ld_x, grad_ld_y = image_gradients(torch.tensor(low_dose_img))
    grad_deno_x, grad_deno_y = image_gradients(torch.tensor(denoised_img))
    grad_hd_x, grad_hd_y = image_gradients(torch.tensor(high_dose_img))

    grad_ld = torch.sqrt(grad_ld_x ** 2 + grad_ld_y ** 2)
    grad_deno = torch.sqrt(grad_deno_x ** 2 + grad_deno_y ** 2)
    grad_hd = torch.sqrt(grad_hd_x ** 2 + grad_hd_y ** 2)

    print(torch.mean(grad_hd), torch.mean(grad_ld), torch.mean(grad_deno))
    print(skew(grad_hd.numpy().flatten()), skew(grad_ld.numpy().flatten()), skew(grad_deno.numpy().flatten()))
    print(kurtosis(grad_hd.numpy().flatten()), kurtosis(grad_ld.numpy().flatten()), kurtosis(grad_deno.numpy().flatten()))
    gradient_difference1 = abs(grad_deno[0, 0, :, :] - grad_ld[0, 0, :, :])
    gradient_difference2 = abs(grad_hd[0, 0, :, :] - grad_ld[0, 0, :, :])
    grad_correlation1 = np.corrcoef(grad_ld_x[0, 0, :, :].flatten(), grad_deno_x[0, 0, :, :].flatten())
    grad_correlation2 = np.corrcoef(grad_ld_x[0, 0, :, :].flatten(), grad_hd_x[0, 0, :, :].flatten())
    print(grad_correlation1, grad_correlation2, np.corrcoef(low_dose_img[0, 0, :, :].flatten(), denoised_img[0, 0, :, :].flatten()), np.corrcoef(low_dose_img[0, 0, :, :].flatten(), high_dose_img[0, 0, :, :].flatten()))"""

    """plt.figure(figsize=(8, 4))
    print(np.max(correlation1), np.max(correlation2))
    plt.subplot(131)
    plt.imshow(grad_ld[0, 0, :, :], cmap='gray')
    plt.subplot(132)
    plt.imshow(grad_deno[0, 0, :, :], cmap='gray')
    plt.subplot(133)
    plt.imshow(grad_hd[0, 0, :, :], cmap='gray')"""

    """plt.figure()
    plt.subplot(241)
    plt.imshow(grad_ld_y[0, 0, :, :], cmap='gray')
    plt.subplot(242)
    plt.imshow(grad_deno_y[0, 0, :, :], cmap='gray')
    plt.subplot(243)
    plt.imshow(grad_hd_y[0, 0, :, :], cmap='gray')
    plt.subplot(244)
    plt.imshow(grad_deno_x[0, 0, :, :], cmap='gray')
    plt.subplot(245)
    plt.imshow(grad_ld_x[0, 0, :, :], cmap='gray')
    plt.subplot(246)
    plt.imshow(grad_deno_x[0, 0, :, :], cmap='gray')
    plt.subplot(247)
    plt.imshow(grad_ld[0, 0, :, :], cmap='gray')
    plt.subplot(248)
    plt.imshow(grad_deno[0, 0, :, :], cmap='gray')
    plt.show()"""

    # kernel_density_estimate(low_dose_img[0, 0, :, :], high_dose_img[0, 0, :, :], denoised_img[0, 0, :, :])
    """plt.imshow(low_dose_img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()
    plt.imshow(denoised_img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()"""

    # for i in zip(x_coord, y_coord):

    # Set the size of the noise template
    #    for t in [32]:
    #        template_size = (t, t)

    # Generate Gaussian noise with mean 127.5 and standard deviation 50
    #        np.random.seed(42)
    #        noise_template = low_dose_img[0, 0, :, :].copy()
    # crop_size = t
    # max_x = noise_template.shape[1] - crop_size
    # max_y = noise_template.shape[0] - crop_size
    # x = np.random.randint(0, max_x)
    # y = np.random.randint(0, max_y)
    # noise_template = noise_template[y:y + crop_size, x:x + crop_size]
    #        noise_template = noise_template[i[0][0]:i[0][1], i[1][0]:i[1][1]]
    # noise_template = np.random.normal(0.0, 0.5, template_size).astype(np.float32)
    #        d = {"c002": {}}
    #        print(d)
    # Clip the noise values to the range [0, 255]
    #        noise_template = np.clip(noise_template, -1.0, 1.0).astype(np.float32)
    #        print(np.min(noise_template), np.max(noise_template))
    #        plt.imshow(noise_template, cmap="gray", vmin=-1, vmax=1)
    #        plt.show()
    #        a = template_matching(low_dose_img, high_dose_img, denoised_img, noise_template)
    #        d = d["c002"]
    #        d = {"c002": {"d": a}}
    #        print(d)
    #        """# Display the noise template
    #        plt.imshow(noise_template, cmap='gray', vmin=0, vmax=255)
    #        plt.axis('off')
    #        plt.show()
    #        print(noise_template.shape)"""

    #        w, h = noise_template.shape[::-1]
    # All the 6 methods for comparison in a list
    #        methods = ['cv.TM_CCOEFF_NORMED']  # 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',, 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'

    #        for meth in methods:

    # img = img2.copy()
    #           method = eval(meth)
    # Apply template Matching
    #            res_1 = cv.matchTemplate(low_dose_img[0, 0, :, :], noise_template, method)
    #            print(np.min(res_1), np.max(res_1))
    #            min_val_1, max_val_1, min_loc_1, max_loc_1 = cv.minMaxLoc(res_1)
    #            print(f"low-dose: {min_val_1, max_val_1}")
    #            res_2 = cv.matchTemplate(denoised_img[0, 0, :, :], noise_template, method)
    #            print(np.min(res_2), np.max(res_2))
    #            min_val_2, max_val_2, min_loc_2, max_loc_2 = cv.minMaxLoc(res_2)
    #            print(f"Denoised: {min_val_2, max_val_2}")
    #            res_3 = cv.matchTemplate(high_dose_img[0, 0, :, :], noise_template, method)
    #            print(np.min(res_3), np.max(res_3))
    #            min_val_3, max_val_3, min_loc_3, max_loc_3 = cv.minMaxLoc(res_3)
    #            print(f"low-dose: {min_val_3, max_val_3}")
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    #            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    #                top_left_1 = min_loc_1
    #                top_left_2 = min_loc_2
    #            else:
    #                top_left_1 = max_loc_1
    #                top_left_2 = max_loc_2
    #                top_left_3 = max_loc_3

    #            ld = low_dose_img.copy()
    #            deno = denoised_img.copy()
    #            hd = high_dose_img.copy()
    #            bottom_right_1 = (top_left_1[0] + w, top_left_1[1] + h)
    #            cv.rectangle(ld[0, 0, :, :], top_left_1, bottom_right_1, 255, 2)

    #            bottom_right_2 = (top_left_2[0] + w, top_left_2[1] + h)
    #            cv.rectangle(deno[0, 0, :, :], top_left_2, bottom_right_2, 255, 2)

    #            bottom_right_3 = (top_left_3[0] + w, top_left_3[1] + h)
    #            cv.rectangle(hd[0, 0, :, :], top_left_3, bottom_right_3, 255, 2)

    #            plt.subplot(161), plt.imshow(res_1, cmap='gray')
    #            plt.title('Matching Result Low-Dose'), plt.xticks([]), plt.yticks([])
    #            plt.subplot(162), plt.imshow(ld[0, 0, :, :], cmap='gray', vmin=-1, vmax=1)
    #            plt.title('Low-Dose'), plt.xticks([]), plt.yticks([])
    #            plt.subplot(163), plt.imshow(res_2, cmap='gray')
    #            plt.title('Matching Result Denoised'), plt.xticks([]), plt.yticks([])
    #            plt.subplot(164), plt.imshow(deno[0, 0, :, :], cmap='gray', vmin=-1, vmax=1)
    #            plt.title('Denoised'), plt.xticks([]), plt.yticks([])
    #            plt.subplot(165), plt.imshow(res_2, cmap='gray')
    #            plt.title('Matching Result High dose'), plt.xticks([]), plt.yticks([])
    #            plt.subplot(166), plt.imshow(hd[0, 0, :, :], cmap='gray', vmin=-1, vmax=1)
    #            plt.title('High Dose'), plt.xticks([]), plt.yticks([])
    #            plt.suptitle(
    #                f"Method: {meth}, Template size: {t}x{t}, \n Average matching map Low-Dose:{np.mean(res_1)} min: {min_val_1}, max: {max_val_1} \n Average matching map denoised:{np.mean(res_2)} min: {min_val_2}, max: {max_val_2}"
    #                f"\n Average matching map high dose:{np.mean(res_3)} min: {min_val_3}, max: {max_val_3}")
    #            plt.show()

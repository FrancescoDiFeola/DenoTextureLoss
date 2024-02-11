from skimage.feature import graycomatrix, graycoprops
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import math


# Define your texture_extractor function as a class
class TextureExtractor(nn.Module):
    def __init__(self, opt):
        super(TextureExtractor, self).__init__()
        self.opt = opt
        if opt.texture_offsets == 'all':
            self.shape = (1, 4, 4)  # 4,4
            self.spatial_offset = [1, 3, 5, 7]
            self.angular_offset = [0, 45, 90, 135]
        elif opt.texture_offsets == '5':
            self.shape = (1, 1, 4)  # 4,4
            self.spatial_offset = [5]
            self.angular_offset = [0]  # 45, 90, 135

    def forward(self, x):
        texture_matrix = torch.empty((x.shape[0], 1, len(self.spatial_offset), len(self.angular_offset)), requires_grad=True)

        # texture_matrix_clone = texture_matrix.clone()
        for i in range(x.shape[0]):
            for idx_d, d in enumerate(self.spatial_offset):
                for idx_theta, theta in enumerate(self.angular_offset):
                    texture_d_theta = compute_haralick_features(
                        glcm_pytorch(x[i].unsqueeze(0), angles=torch.tensor([theta * math.pi / 180]).to(torch.float32), distances=torch.tensor([d]).to(torch.float32), levels=256))
                    texture_matrix[i, 0, idx_d - 1, idx_theta - 1] = texture_d_theta

        return texture_matrix


class TextureExtractor1(nn.Module):
    def __init__(self, opt):
        super(TextureExtractor1, self).__init__()
        self.opt = opt
        if opt.texture_offsets == 'all':
            self.shape = (1, 4, 4)  # 4, 4
            self.spatial_offset = [1, 3, 5, 7]
            self.angular_offset = [0, 45, 90, 135]
        elif opt.texture_offsets == '5':
            self.shape = (1, 1, 4)  # 4, 4
            self.spatial_offset = [5]
            self.angular_offset = [0, 45, 90, 135]

    def forward(self, x):

        list = []
        for i in range(x.shape[0]):
            for idx_d, d in enumerate(self.spatial_offset):
                for idx_theta, theta in enumerate(self.angular_offset):
                    if d == 5 and theta == 0:
                        kernel = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               ], dtype=torch.float32, requires_grad=True)
                    elif d == 5 and theta == 45:
                        kernel = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               ], dtype=torch.float32, requires_grad=True)
                    elif d == 5 and theta == 90:
                        kernel = torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               ], dtype=torch.float32, requires_grad=True)
                    elif d == 5 and theta == 135:
                        kernel = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                               ], dtype=torch.float32, requires_grad=True)
                    texture_d_theta = compute_haralick_features(convolutional_glcm(x[i, 0, :, :], kernel, 16, d))
                    list.append(texture_d_theta.unsqueeze(0))

        texture_matrix = torch.cat(list, dim=0).view(x.shape[0], 1, len(self.spatial_offset), len(self.angular_offset))
        return texture_matrix


# Not differentiable
def _texture_extractor(x, opt):
    x = cv2.normalize(x.detach().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                      dtype=cv2.CV_32F).astype(np.uint8)

    if opt.texture_offsets == 'all':
        shape = (x.shape[0], 1, 4, 4)  # 4,4
        spatial_offset = [1, 3, 5, 7]
        angular_offset = [0, 45, 90, 135]
    elif opt.texture_offsets == '5':
        shape = (x.shape[0], 1, 1, 4)  # 4,4
        spatial_offset = [5]
        angular_offset = [0, 45, 90, 135]

    texture_matrix = torch.empty(shape)
    texture_matrix_clone = texture_matrix.clone()
    for i in range(0, x.shape[0]):
        for idx_d, d in enumerate(spatial_offset):
            for idx_theta, theta in enumerate(angular_offset):
                texture_d_theta = graycoprops(
                    graycomatrix(x[i, 0, :, :], distances=[d], angles=[theta], levels=256, symmetric=True,
                                 normed=True), "contrast")[0][0]
                texture_matrix[i, 0, idx_d - 1, idx_theta - 1] = texture_d_theta

    return texture_matrix


# Differentiable Haralicks
def compute_haralick_features(glcm):
    glcm = glcm.permute(2, 3, 1, 0)
    num_levels = glcm.shape[0]

    # Compute probabilities for each (i, j) pair
    p_ij = glcm.view(num_levels * num_levels, -1).sum(dim=1)
    # Compute means and variances
    i_indices, j_indices = torch.meshgrid(torch.arange(num_levels), torch.arange(num_levels))

    i_indices = i_indices.reshape(-1).to(torch.float32)
    j_indices = j_indices.reshape(-1).to(torch.float32)
    # mean_i = (i_indices * p_ij).sum()
    # mean_j = (j_indices * p_ij).sum()
    # var_i = ((i_indices - mean_i) ** 2 * p_ij).sum()
    # var_j = ((j_indices - mean_j) ** 2 * p_ij).sum()

    # Compute Haralick features
    contrast = ((i_indices - j_indices) ** 2 * p_ij).sum()
    # correlation = ((i_indices - mean_i) * (j_indices - mean_j) * p_ij).sum() / (var_i * var_j).sqrt()
    # energy = (p_ij ** 2).sum()  # Square the probabilities for energy
    # homogeneity = (p_ij / (1.0 + torch.abs(i_indices - j_indices))).sum()

    # haralick_features = torch.tensor([contrast, correlation, energy, homogeneity])

    return contrast


# logic GLCM
def logic_glcm(image, levels, normalize=False):
    image = torch.floor(image * levels) / levels + 1
    print(image)
    L_i = []
    Ls_i = []
    glcm = []

    for i in range(1, levels + 1):
        L_i.append(image * (image == float(i)))
        t = torch.roll(image * (image == float(i)), shifts=1, dims=3)

        t[:, :, :, 0] = 0
        print(t)
        Ls_i.append(t)

    for j in range(1, levels + 1):
        for k in range(1, levels + 1):
            # print(f"pair {j}-{k}:")
            # print(f"Ls_i: {Ls_i[j-1]}")
            # print(f"L_i: {L_i[k-1]}")
            # print(f"rij: {torch.logical_and(Ls_i[j-1], L_i[k-1])}")

            glcm.append((Ls_i[j - 1] * L_i[k - 1]).sum().unsqueeze(0))  # torch.logical_and(Ls_i[j-1], L_i[k-1])

    glcm = torch.cat(glcm, dim=0).view(1, 1, levels, levels)
    # glcm = glcm+glcm.transpose(2, 3)

    if normalize:
        # Normalize each GLCM
        glcm_sum = torch.sum(glcm, dim=(2, 3), keepdim=True)
        # glcm_sum[glcm_sum == 0] = 1
        glcm /= glcm_sum

    # print(f"GLCM: {glcm}")
    return glcm


# custom GLCM with convolutions
def convolutional_glcm(image, kernel, levels, pad_value, normalize=True):
    # Map to range (1, levels)
    image = ((levels - 1) * (image - image.min()) / (image.max() - image.min())) + 1

    # print("image: ")
    # print(image)
    # Compute the GLCM using convolutions
    list_glcm = []
    for i in range(1, levels + 1):
        for j in range(1, levels + 1):
            specified_pair = [i, j]

            # Apply the mask to replace masked-out values with zeros
            mask = torch.logical_or(image.unsqueeze(0).unsqueeze(0) == specified_pair[0], image.unsqueeze(0).unsqueeze(0) == specified_pair[1])
            masked_tensor = torch.where(mask, image, 0)  # gradient on
            padded_input = F.pad(masked_tensor, (pad_value, pad_value, pad_value, pad_value), value=20)  # gradient on
            # print(f"Padded input pair {i}-{j}: {padded_input}")
            result = F.conv2d(padded_input, kernel.unsqueeze(0).unsqueeze(0))  # gradient on
            masked_result = (result * (torch.abs(result - (i + j)) < 1e-6))
            # print(f"Masked result pair {i}-{j}: {masked_result}")

            if masked_result.sum() == 0:
                # print(f"n.occ pair {i}-{j}:{(masked_result[0, 0, 0, 0]).unsqueeze(0)}")
                list_glcm.append((masked_result[0, 0, 0, 0]).unsqueeze(0))
            else:
                non_zero = (result * (torch.abs(result - (i + j)) < 1e-6)).nonzero()
                masked_value = masked_result[non_zero[0, 0], non_zero[0, 1], non_zero[0, 2], non_zero[0, 3]]
                # print(f"Masked value pair {i}-{j}:{masked_value}")
                if i == j:
                    list_glcm.append((2 * (masked_result.sum() / masked_value)).unsqueeze(0))  # 2 * (((result - (i + j)) == 0).sum().unsqueeze(0).float().requires_grad_(True))
                    # print(f"n occ. pair {i}-{j}:{2 * (masked_result.sum() / masked_value).unsqueeze(0)}")
                else:
                    list_glcm.append((masked_result.sum() / masked_value).unsqueeze(0))  # ((result - (i + j)) == 0).sum().unsqueeze(0).float().requires_grad_(True)
                    # print(f"n occ. pair {i}-{j}:{(masked_result.sum() / masked_value).unsqueeze(0)}")

    glcm = torch.cat(list_glcm, dim=0).view(1, 1, levels, levels)

    if normalize:
        # Normalize each GLCM
        glcm_sum = torch.sum(glcm, dim=(2, 3), keepdim=True)
        # glcm_sum[glcm_sum == 0] = 1  # Avoid division by zero
        glcm /= glcm_sum
    # print(f"GLCM: {glcm}")
    return glcm


# Differentiable GLCM (Gray-Level-Cooccurence matirx) (torch-based implementation)
def glcm_pytorch(image, angles, distances, levels, symmetric=True, normalize=True):
    """Perform co-occurrence matrix accumulation.
    Parameters
    ----------
    image : torch.tensor of shape (B, C, W, H),
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    angles : torch.tensor of data type int and shape (aa, )
        List of pixel pair angles in radians.
    distances : torch.tensor of data type int and shape (dd, )
        List of pixel pair distance offsets.

    levels : int
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image).
    returns
    out : torch.tensor
        On input a 6D tensor of shape (B, C, levels, levels, aa, dd) and integer values
        that returns the results of the GLCM computation.
    """
    # The following check can be done in the python front end:
    if torch.sum((image >= 0) & (image < levels)).item() < 1:
        raise ValueError("image values cannot exceed levels and also must be positive!!")

    # Map to range (0, levels-1)
    image = ((levels - 1) * (image - image.min()) / (image.max() - image.min()))

    batch_size = image.size(0)
    c_in = image.size(1)
    rows = image.size(2)
    cols = image.size(3)
    aa = angles.size(0)
    dd = distances.size(0)
    out_o = torch.zeros((levels, levels, aa, dd), dtype=torch.float32, requires_grad=True)

    out = out_o.clone()
    angles_mesh, distances_mesh = torch.meshgrid(angles, distances)

    offset_row = torch.round(torch.sin(angles_mesh) * distances_mesh).long()
    offset_col = torch.round(torch.cos(angles_mesh) * distances_mesh).long()
    start_row = torch.where(offset_row > 0, 0, -offset_row)

    end_row = torch.where(offset_row > 0, rows - offset_row, rows)
    start_col = torch.where(offset_col > 0, 0, -offset_col)
    end_col = torch.where(offset_col > 0, cols - offset_col, cols)

    for a_idx in range(angles.size(0)):
        for d_idx in range(distances.size(0)):
            rs0 = start_row[a_idx, d_idx]
            cs0 = start_col[a_idx, d_idx]
            rs1 = rs0 + offset_row[a_idx, d_idx]
            cs1 = cs0 + offset_col[a_idx, d_idx]

            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    # compute the location of the offset pixel
                    row = r + rs1
                    col = c + cs1

                    out[int(image[:, :, r, c]), int(image[:, :, row, col]), a_idx, d_idx] += 1

    # make each GLMC symmetric
    if symmetric:
        Pt = out_o.permute(1, 0, 2, 3)
        out = out + Pt

    if normalize:
        # Normalize each GLCM
        out_sum = torch.sum(out, dim=(0, 1), keepdim=True)
        out_sum[out_sum == 0] = 1  # Avoid division by zero
        out /= out_sum
    print(out.shape)
    return out_o


# Differentiable GLCM (Gray-Level-Cooccurence matirx) (torch-based implementation)
def glcm_pytorch2(image, angles, distances, levels, symmetric=True, normalize=True):
    """Perform co-occurrence matrix accumulation.
    Parameters
    ----------
    image : torch.tensor of shape (B, C, W, H),
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    angles : torch.tensor of data type int and shape (aa, )
        List of pixel pair angles in radians.
    distances : torch.tensor of data type int and shape (dd, )
        List of pixel pair distance offsets.

    levels : int
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image).
    returns
    out : torch.tensor
        On input a 6D tensor of shape (B, C, levels, levels, aa, dd) and integer values
        that returns the results of the GLCM computation.
    """
    # The following check can be done in the python front end:
    if torch.sum((image >= 0) & (image < levels)).item() < 1:
        raise ValueError("image values cannot exceed levels and also must be positive!!")

    # Map to range (0, levels-1)
    image = ((levels - 1) * (image - image.min()) / (image.max() - image.min()))

    batch_size = image.size(0)
    c_in = image.size(1)
    rows = image.size(2)
    cols = image.size(3)
    aa = angles.size(0)
    dd = distances.size(0)
    out_o = torch.zeros((levels, levels, aa, dd), dtype=torch.float32, requires_grad=True)

    # out = out_o.clone()
    angles_mesh, distances_mesh = torch.meshgrid(angles, distances)

    offset_row = torch.round(torch.sin(angles_mesh) * distances_mesh).long()
    offset_col = torch.round(torch.cos(angles_mesh) * distances_mesh).long()
    start_row = torch.where(offset_row > 0, 0, -offset_row)
    end_row = torch.where(offset_row > 0, rows - offset_row, rows)
    start_col = torch.where(offset_col > 0, 0, -offset_col)
    end_col = torch.where(offset_col > 0, cols - offset_col, cols)

    for a_idx in range(angles.size(0)):
        for d_idx in range(distances.size(0)):
            rs0 = start_row[a_idx, d_idx]
            cs0 = start_col[a_idx, d_idx]
            rs1 = rs0 + offset_row[a_idx, d_idx]
            cs1 = cs0 + offset_col[a_idx, d_idx]

            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    # compute the location of the offset pixel
                    row = r + rs1
                    col = c + cs1

                    out_o[int(image[:, :, r, c]), int(image[:, :, row, col]), a_idx, d_idx] += 1

    # make each GLMC symmetric
    if symmetric:
        Pt = out_o.permute(1, 0, 2, 3)
        out_o = out_o + Pt

    if normalize:
        # Normalize each GLCM
        out_sum = torch.sum(out_o, dim=(0, 1), keepdim=True)
        out_sum[out_sum == 0] = 1  # Avoid division by zero
        out_o /= out_sum

    return out_o


def glcm_pytorch1(image, angles, distances, levels, symmetric=True, normalize=True):
    if torch.sum((image >= 0) & (image < levels)).item() < 1:
        raise ValueError("Image values cannot exceed levels and also must be positive!!")

    image = ((levels - 1) * (image - image.min()) / (image.max() - image.min()))

    batch_size, c_in, rows, cols = image.size()
    aa, dd = angles.size(0), distances.size(0)
    out_o = torch.zeros((levels, levels, aa, dd), dtype=torch.float32, requires_grad=True)

    out = out_o.clone()
    angles_mesh, distances_mesh = torch.meshgrid(angles, distances)

    offset_row = torch.round(torch.sin(angles_mesh) * distances_mesh).long()
    offset_col = torch.round(torch.cos(angles_mesh) * distances_mesh).long()
    start_row = torch.where(offset_row > 0, 0, -offset_row)
    end_row = torch.where(offset_row > 0, rows - offset_row, rows)
    start_col = torch.where(offset_col > 0, 0, -offset_col)
    end_col = torch.where(offset_col > 0, cols - offset_col, cols)

    r_range = torch.arange(rows).unsqueeze(0)
    c_range = torch.arange(cols).unsqueeze(0)

    for a_idx in range(angles.size(0)):
        for d_idx in range(distances.size(0)):
            rs0 = start_row[a_idx, d_idx]
            cs0 = start_col[a_idx, d_idx]
            rs1 = rs0 + offset_row[a_idx, d_idx]
            cs1 = cs0 + offset_col[a_idx, d_idx]

            r = r_range.unsqueeze(2) + rs1
            c = c_range.unsqueeze(2) + cs1

            valid = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
            r, c = r[valid], c[valid]
            src = image[:, :, r, c]
            dst = image[:, :, r_range, c_range]
            print(int(src))
            print(dst)
            # print(int(a_idx))
            # print(int(d_idx))
            out[src, dst, a_idx, d_idx] += valid.int()

    if symmetric:
        Pt = out.permute(1, 0, 2, 3)
        out = out + Pt

    if normalize:
        out_sum = torch.sum(out, dim=(0, 1), keepdim=True)
        out_sum[out_sum == 0] = 1  # Avoid division by zero
        out /= out_sum

    return out
def texture_extractor(x, opt):
    x = cv2.normalize(x.detach().cpu().numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                      dtype=cv2.CV_32F).astype(np.uint8)

    if opt.texture_offsets == 'all':
        shape = (x.shape[0], 1, 4, 4)  # 4,4
        spatial_offset = [1, 3, 5, 7]
        angular_offset = [0, 45, 90, 135]
    elif opt.texture_offsets == '5':
        shape = (x.shape[0], 1, 1, 4)  # 4,4
        spatial_offset = [5]
        angular_offset = [0, 45, 90, 135]

    texture_matrix = torch.empty(shape)

    for i in range(0, x.shape[0]):
        for idx_d, d in enumerate(spatial_offset):
            for idx_theta, theta in enumerate(angular_offset):
                texture_d_theta = graycoprops(
                    graycomatrix(x[i, 0, :, :], distances=[d], angles=[theta], levels=256, symmetric=True,
                                 normed=True), "contrast")[0][0]
                texture_matrix[i, 0, idx_d - 1, idx_theta - 1] = texture_d_theta

    return texture_matrix
def texture_loss2(fake_im, real_im, operator, opt, model=None):
    textures_real = texture_extractor(real_im, opt)
    textures_fake = texture_extractor(fake_im, opt)

    if opt.texture_criterion == 'attention':
        criterion = operator(textures_fake, textures_real).view(opt.batch_size, 1, 4, 4)
        normalized_criterion = (criterion - criterion.min()) / (criterion.max() - criterion.min())
        print(normalized_criterion.shape)
        out_attention, map, weight = model(normalized_criterion)
        print(out_attention.shape)
        loss_cycle_texture = abs(torch.mean(torch.sum(out_attention, dim=(2, 3))))

        return loss_cycle_texture, map, weight

    elif opt.texture_criterion == 'max':
        delta_grids = operator(textures_fake, textures_real).view(opt.batch_size, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture, _ = torch.max(delta_grids, dim=1)  # return a tensor 1xB

        # compute the loss function by averaging over the batch
        loss_cycle_texture = torch.mean(criterion_texture)
        return loss_cycle_texture, delta_grids, criterion_texture

    elif opt.texture_criterion == 'average':
        delta_grids = operator(textures_fake, textures_real).view(opt.batch_size, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture = torch.mean(delta_grids, dim=1)  # return a tensor 1xB
        loss_cycle_texture = torch.mean(criterion_texture)

        return loss_cycle_texture

    elif opt.texture_criterion == 'Frobenius':
        delta_grids = operator(textures_fake, textures_real).view(opt.batch_size, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture = frobenius_dist(delta_grids)  # return a tensor 1xB
        loss_cycle_texture = torch.mean(criterion_texture)

        return loss_cycle_texture



def texture_loss(fake_im, real_im, operator, opt, extractor, model=None):
    textures_real = extractor(real_im)  # texture_
    textures_fake = extractor(fake_im)  # texture_

    if opt.texture_criterion == 'attention':
        criterion = operator(textures_fake, textures_real).view(opt.batch_size, 1, 4, 4)
        normalized_criterion = (criterion - criterion.min()) / (criterion.max() - criterion.min())
        out_attention, map, weight = model(normalized_criterion)
        loss_cycle_texture = abs(torch.mean(torch.sum(out_attention, dim=(2, 3))))

        return loss_cycle_texture, map, weight

    elif opt.texture_criterion == 'max':
        delta_grids = operator(textures_fake, textures_real).view(opt.batch_size, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture, _ = torch.max(delta_grids, dim=1)  # return a tensor 1xB

        # compute the loss function by averaging over the batch
        loss_cycle_texture = torch.mean(criterion_texture)
        return loss_cycle_texture, delta_grids, criterion_texture

    elif opt.texture_criterion == 'average':
        delta_grids = operator(textures_fake, textures_real)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture = torch.mean(delta_grids, dim=1)  # return a tensor 1xB
        loss_cycle_texture = torch.mean(criterion_texture)
        return loss_cycle_texture

    elif opt.texture_criterion == 'Frobenius':
        delta_grids = operator(textures_fake, textures_real).view(opt.batch_size, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture = frobenius_dist(delta_grids)  # return a tensor 1xB
        loss_cycle_texture = torch.mean(criterion_texture)

        return loss_cycle_texture


def frobenius_dist(t1):
    dot_prod = t1 * t1
    return torch.sqrt(torch.sum(dot_prod, dim=1))


def calculate_glcm_bool(image, d=1, theta=0, levels=256):
    # Convert the input image to grayscale if it's not already.
    if torch.all(image == 0):
        return torch.zeros((levels, levels), dtype=torch.float32).to(image.device)

    # Normalize the image to the specified number of levels.
    image = (image * (levels - 1)).clamp(0, levels - 1).round().long()

    # Calculate the pixel offsets based on the given distance and angle.
    dx = d * torch.cos(torch.deg2rad(theta))
    dy = d * torch.sin(torch.deg2rad(theta))

    # Convert dx and dy to integers for shifts
    dx_int = int(dx)
    dy_int = int(dy)

    # Shift the image using the calculated offsets.
    image_shifted = torch.roll(image, shifts=(0, 1), dims=(0, 1))
    print(image_shifted)
    # Calculate the GLCM matrix.
    glcm = torch.zeros((levels, levels), dtype=torch.float32).to(image.device)
    for i in range(levels):
        for j in range(levels):
            mask = (image == i) & (image_shifted == j)
            glcm[i, j] = mask.sum().float()

    # Normalize the GLCM to have a sum of 1.
    # glcm /= glcm.sum()

    return glcm


def _glcm_loop_torch(image, angles, distances, levels, symmetric=True, normalize=True):
    """Perform co-occurrence matrix accumulation.
    Parameters
    ----------
    image : torch.tensor of shape (B, C, W, H),
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    angles : torch.tensor of data type int and shape (aa, )
        List of pixel pair angles in radians.
    distances : torch.tensor of data type int and shape (dd, )
        List of pixel pair distance offsets.

    levels : int
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image).
    returns
    out : torch.tensor
        On input a 6D tensor of shape (B, C, levels, levels, aa, dd) and integer values
        that returns the results of the GLCM computation.
    """
    # The following check can be done in the python front end:
    if torch.sum((image >= 0) & (image < levels)).item() < 1:
        raise ValueError("image values cannot exceed levels and also must be positive!!")

    # Map to range (0, levels-1)
    # image = ((levels - 1) * (image - image.min()) / (image.max() - image.min()))

    batch_size = image.size(0)
    c_in = image.size(1)
    rows = image.size(2)
    cols = image.size(3)
    aa = angles.size(0)
    dd = distances.size(0)
    out_o = torch.zeros((levels, levels, aa, dd), dtype=torch.float32, requires_grad=True)

    out = out_o.clone()
    angles_mesh, distances_mesh = torch.meshgrid(angles, distances, indexing="ij")

    offset_row = torch.round(torch.sin(angles_mesh) * distances_mesh).long()
    offset_col = torch.round(torch.cos(angles_mesh) * distances_mesh).long()
    start_row = torch.where(offset_row > 0, 0, -offset_row)
    end_row = torch.where(offset_row > 0, rows - offset_row, rows)
    start_col = torch.where(offset_col > 0, 0, -offset_col)
    end_col = torch.where(offset_col > 0, cols - offset_col, cols)

    for a_idx in range(angles.size(0)):
        for d_idx in range(distances.size(0)):
            rs0 = start_row[a_idx, d_idx]
            cs0 = start_col[a_idx, d_idx]
            rs1 = rs0 + offset_row[a_idx, d_idx]
            cs1 = cs0 + offset_col[a_idx, d_idx]

            for r in range(start_row, end_row):
                for c in range(start_col, end_col):
                    # compute the location of the offset pixel
                    row = r + rs1
                    col = c + cs1

                    out[int(image[:, :, r, c]), int(image[:, :, row, col]), a_idx, d_idx] += 1

    # make each GLMC symmetric
    if symmetric:
        Pt = out.permute(1, 0, 2, 3)
        out = out + Pt

    if normalize:
        # Normalize each GLCM
        out_sum = torch.sum(out, dim=(0, 1), keepdim=True)
        out_sum[out_sum == 0] = 1  # Avoid division by zero
        out /= out_sum

    return out


if __name__ == "__main__":
    import math

    # image = torch.randint(0, 2, (1, 1, 3, 3))

    image2 = torch.tensor([[[[0, 0, 1],
                             [0, 0, 1],
                             [0, 1, 0]]]])
    """image2 = torch.tensor([[[[1, 1, 2],
                             [1, 1, 2],
                             [1, 2, 1]]],
                            [[[2, 3, 2],
                             [1, 1, 2],
                             [1, 2, 1]]]], dtype=torch.float32)"""
    # print(torch.roll(image2, shifts=1, dims=3))
    print(image2.shape)
    print(f"logic glcm: {logic_glcm(image2, 3)}")
    image2 = torch.tensor([[[[0, 0, 1],
                             [0, 0, 1],
                             [0, 1, 0]]]], dtype=torch.uint8)
    print(f"standard glcm:{graycomatrix(image2[2, 0, :, :].detach().numpy(), distances=[1], angles=[0], levels=3, symmetric=True, normed=False)}")


    def compute_glcm(image, distance=1, angles=[0], levels=3):
        # Convert the image to a PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32)

        # Compute co-occurrence matrix
        glcm = torch.zeros((levels, levels), dtype=torch.float32)

        for angle in angles:
            shifted = F.conv2d(image, torch.ones(1, 1, 2 * distance + 1, 2 * distance + 1).to(image.device), padding=distance, stride=1)
            shifted = shifted.clamp(min=0, max=levels - 1).to(torch.int64)

            # Flatten the shifted tensor and compute bincount
            flat_shifted = shifted.view(-1)

            # Compute the weights tensor to be used with bincount
            weights = torch.ones_like(flat_shifted)

            glcm += torch.bincount(levels * flat_shifted + flat_shifted, weights, levels ** 2).view(levels, levels)

        # Normalize GLCM
        # glcm /= glcm.sum()

        return glcm


    angles = torch.tensor([0 * (math.pi / 180)]).to(torch.float32)
    # angles = torch.tensor([0]).to(torch.float32)
    distances = torch.tensor([2]).to(torch.float32)

    out = glcm_pytorch(image2, angles, distances, levels=3, symmetric=True, normalize=False)

    image2 = torch.tensor([[[[0, 0, 2],
                             [0, 0, 2],
                             [0, 2, 0]]]])
    """image2 = torch.tensor([[[[0, 0, 1],
                          [0, 0, 1],
                          [0, 1, 0]]]])"""
    # print(graycomatrix(image2[0, 0, :, :].detach().numpy(), distances=[2], angles=[0 * (math.pi / 180)], levels=3, symmetric=True, normed=False))

    """lista_1 = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    lista_2 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    for j in zip(lista_1, lista_2):
        for i in zip(lista_1, lista_2):
            print((i[0]*j[1])+(i[1]*j[1]))"""

    # lista_1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 2, 2, 2, 3, 3, 3]
    # lista_2 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    # Load your grayscale image as a PyTorch tensor
    """image = torch.tensor([[1, 1, 2],
                          [2, 1, 2],
                          [3, 2, 1]], dtype=torch.float32, requires_grad=True)"""
    image = torch.tensor([[0, 0, 1],
                          [1, 0, 1],
                          [2, 1, 0]], dtype=torch.float32, requires_grad=True)

    image2 = torch.tensor([[0, 0, 1],
                           [1, 0, 1],
                           [2, 1, 0]], dtype=torch.uint8)

    image3 = torch.tensor([[0, 0, 1, 0, 0, 1],
                           [1, 0, 1, 1, 0, 1],
                           [2, 1, 0, 2, 1, 0],
                           [2, 1, 0, 2, 1, 0],
                           [2, 1, 0, 2, 1, 0],
                           [2, 1, 0, 2, 1, 0]], dtype=torch.uint8)

    image4 = torch.tensor([[1, 1, 2, 1, 1, 2],
                           [2, 1, 2, 2, 1, 2],
                           [3, 2, 1, 3, 2, 1],
                           [3, 2, 1, 3, 2, 1],
                           [3, 2, 1, 3, 2, 1],
                           [3, 2, 1, 3, 2, 1]], dtype=torch.float32)

    print(graycomatrix(image2.detach().numpy(), distances=[1], angles=[0], levels=3, symmetric=False, normed=True))
    # print(calculate_glcm_bool(image2, torch.tensor(1), torch.tensor(0), 3))
    # Define the number of levels in the GLCM
    levels = 3

    # Initialize an empty GLCM
    # glcm = torch.zeros(levels, levels, dtype=torch.float32)

    # Create custom convolutional kernels for co-occurrence pairs (kernel size: (2d+1)x(2d+1))
    # d = 1, theta = 0
    kernel = torch.tensor([[0, 0, 0],
                           [0, 1, 1],
                           [0, 0, 0]], dtype=torch.float32, requires_grad=True)
    """kernels = []
    for i in range(1, levels + 1):
        for j in range(1, levels + 1):
            kernel = torch.tensor([[0, 0, 0],
                                   [0, 1 , 1],
                                   [0, 0, 0]], dtype=torch.float32)
            kernels.append(kernel)"""

    # d = 2, theta = 0
    """kernels = []
    for i in range(1, levels + 1):
        for j in range(1, levels + 1):
            kernel = torch.tensor([[0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 1],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0]], dtype=torch.float32)
            kernels.append(kernel)"""
    """kernel = torch.tensor([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 1],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]], dtype=torch.float32)"""
    # d = 3, theta = 0
    '''kernels = []
    for i in range(1, levels + 1):
        for j in range(1, levels + 1):
            kernel = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1 / int(i), 0, 0, 1 / int(j)],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
            kernels.append(kernel)'''

    # padding d = 1
    # padded_input = F.pad(image, (1, 1, 1, 1), value=20)
    # padding d = 2
    # padded_input = F.pad(image, (2, 2, 2, 2), value=1000)
    # print(padded_input)
    # padding d = 3
    # padded_input = F.pad(image4, (3, 3, 3, 3), value=1000)
    # print(padded_input)

    # Compute the GLCM using convolutions and accumulate the results
    """c = 0
    list_glcm = []
    # glcm = torch.empty((1, 1, levels, levels), requires_grad=True)
    pad_value = 1
    for i in range(1, levels + 1):
        for j in range(1, levels + 1):
            specified_pair = [i, j]
            print(f"pair: {i}, {j}")
            # Apply the mask to replace masked-out values with zeros
            mask = torch.logical_or(image.unsqueeze(0).unsqueeze(0) == specified_pair[0], image.unsqueeze(0).unsqueeze(0) == specified_pair[1])
            masked_tensor = torch.where(mask, image, 0)  # gradient on
            padded_input = F.pad(masked_tensor, (pad_value, pad_value, pad_value, pad_value), value=20)  # gradient on
            print(f"padded input: {padded_input}")

            result = F.conv2d(padded_input, kernel.unsqueeze(0).unsqueeze(0))  # gradient on
            print(f"Result  convolution pair {i}-{j}:")
            print(result)

            if i == j:
                list_glcm.append(2 * (((result - (i + j)) == 0).sum().unsqueeze(0).float().requires_grad_(True)))
            else:
                list_glcm.append(((result - (i + j)) == 0).sum().unsqueeze(0).float().requires_grad_(True))

            c += 1
    glcm = torch.cat(list_glcm, dim=0).view(1, 1, levels, levels)

    print(glcm)"""
    print("---------")
    pad_value = 1
    glcm = convolutional_glcm(image, kernel, levels, pad_value, normalize=True)
    print(glcm)
    print(compute_haralick_features(glcm))

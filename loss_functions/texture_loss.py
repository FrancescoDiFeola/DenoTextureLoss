from skimage.feature import graycomatrix, graycoprops
import torch
import cv2
import numpy as np


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


def texture_loss(rec_A, rec_B, real_A, real_B, operator, opt, attention=None):
    textures_real_A = texture_extractor(real_A, opt)
    textures_rec_A = texture_extractor(rec_A, opt)
    textures_real_B = texture_extractor(real_B, opt)
    textures_rec_B = texture_extractor(rec_B, opt)

    if opt.texture_criterion == 'attention':
        criterion_A = operator(textures_rec_A, textures_real_A).view(2, 1, 4, 4)
        normalized_criterion_A = (criterion_A - criterion_A.min()) / (criterion_A.max() - criterion_A.min())
        criterion_B = operator(textures_rec_B, textures_real_B).view(2, 1, 4, 4)
        normalized_criterion_B = (criterion_B - criterion_B.min()) / (criterion_B.max() - criterion_B.min())
        out_attention_A, map_A, weight_A = attention(normalized_criterion_A)
        out_attention_B, map_B, weight_B = attention(normalized_criterion_B)

        loss_cycle_texture_A = torch.sum(torch.sum(out_attention_A, dim=(2, 3)))
        loss_cycle_texture_B = torch.sum(torch.sum(out_attention_B, dim=(2, 3)))

        return loss_cycle_texture_A, loss_cycle_texture_B, map_A, map_B, weight_A, weight_B

    elif opt.texture_criterion == 'max':

        delta_grids_A = operator(textures_rec_A, textures_real_A).view(2, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture_A, _ = torch.max(delta_grids_A, dim=1)  # return a tensor 1xB

        delta_grids_B = operator(textures_rec_B, textures_real_B).view(2, -1)
        criterion_texture_B, _ = torch.max(delta_grids_B, dim=1)

        # compute the loss function by averaging over the batch
        loss_cycle_texture_A = torch.mean(criterion_texture_A)
        loss_cycle_texture_B = torch.mean(criterion_texture_B)

        return loss_cycle_texture_A, loss_cycle_texture_B, delta_grids_A, criterion_texture_A, delta_grids_B, criterion_texture_B

    elif opt.texture_criterion == 'average':
        delta_grids_A = operator(textures_rec_A, textures_real_A).view(2, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture_A = torch.mean(delta_grids_A, dim=1)  # return a tensor 1xB

        delta_grids_B = operator(textures_rec_B, textures_real_B).view(2, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture_B = torch.mean(delta_grids_B, dim=1)  # return a tensor 1xB

        loss_cycle_texture_A = torch.mean(criterion_texture_A)
        loss_cycle_texture_B = torch.mean(criterion_texture_B)

        return loss_cycle_texture_A, loss_cycle_texture_B

    elif opt.texture_criterion == 'Frobenius':
        delta_grids_A = operator(textures_rec_A, textures_real_A).view(2, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture_A = frobenius_dist(delta_grids_A)  # return a tensor 1xB

        delta_grids_B = operator(textures_rec_B, textures_real_B).view(2, -1)  # change shape from BxCxHxW to Bx(HXW)
        criterion_texture_B = frobenius_dist(delta_grids_B)  # return a tensor 1xB

        loss_cycle_texture_A = torch.mean(criterion_texture_A)
        loss_cycle_texture_B = torch.mean(criterion_texture_B)

        return loss_cycle_texture_A, loss_cycle_texture_B


def frobenius_dist(t1):
    dot_prod = t1 * t1
    return torch.sqrt(torch.sum(dot_prod, dim=1))
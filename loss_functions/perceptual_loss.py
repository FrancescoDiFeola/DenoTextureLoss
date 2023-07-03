from models.vgg import VGG
from surgeon_pytorch import Inspect
import torch


def perceptual_similarity_loss(rec_A, rec_B, real_A, real_B, model, perceptual_layers):
    rec_A = rec_A.expand(-1, 3, -1, -1)
    rec_B = rec_B.expand(-1, 3, -1, -1)
    real_A = real_A.expand(-1, 3, -1, -1)
    real_B = real_B.expand(-1, 3, -1, -1)

    f1_real_A, f2_real_A, f3_real_A, f4_real_A, f5_real_A = vgg_16_pretrained(
        real_A, model)  # extract features from vgg
    f1_real_B, f2_real_B, f3_real_B, f4_real_B, f5_real_B = vgg_16_pretrained(real_B, model)

    f1_rec_A, f2_rec_A, f3_rec_A, f4_rec_A, f5_rec_A = vgg_16_pretrained(
        rec_A, model)  # extract features from vgg (1st,2nd, 3rd, 4th, 5th pool)
    f1_rec_B, f2_rec_B, f3_rec_B, f4_rec_B, f5_rec_B = vgg_16_pretrained(rec_B, model)

    if  perceptual_layers == 'all':
        m1_A = torch.mean(torch.mean((f1_real_A - f1_rec_A) ** 2))  # mse difference
        m2_A = torch.mean(torch.mean((f2_real_A - f2_rec_A) ** 2))
        m3_A = torch.mean(torch.mean((f3_real_A - f3_rec_A) ** 2))
        m4_A = torch.mean(torch.mean((f4_real_A - f4_rec_A) ** 2))
        m5_A = torch.mean(torch.mean((f5_real_A - f5_rec_A) ** 2))

        m1_B = torch.mean(torch.mean((f1_real_B - f1_rec_B) ** 2))
        m2_B = torch.mean(torch.mean((f2_real_B - f2_rec_B) ** 2))
        m3_B = torch.mean(torch.mean((f3_real_B - f3_rec_B) ** 2))
        m4_B = torch.mean(torch.mean((f4_real_B - f4_rec_B) ** 2))
        m5_B = torch.mean(torch.mean((f5_real_B - f5_rec_B) ** 2))

        perceptual_loss_A = m1_A + m2_A + m3_A + m4_A + m5_A
        perceptual_loss_B = m1_B + m2_B + m3_B + m4_B + m5_B
    elif perceptual_layers == '1-2':
        m1_A = torch.mean(torch.mean((f1_real_A - f1_rec_A) ** 2))  # mse difference
        m2_A = torch.mean(torch.mean((f2_real_A - f2_rec_A) ** 2))

        m1_B = torch.mean(torch.mean((f1_real_B - f1_rec_B) ** 2))
        m2_B = torch.mean(torch.mean((f2_real_B - f2_rec_B) ** 2))
        perceptual_loss_A = m1_A + m2_A
        perceptual_loss_B = m1_B + m2_B
    elif perceptual_layers == '4-5':
        m4_A = torch.mean(torch.mean((f4_real_A - f4_rec_A) ** 2))
        m5_A = torch.mean(torch.mean((f5_real_A - f5_rec_A) ** 2))

        m4_B = torch.mean(torch.mean((f4_real_B - f4_rec_B) ** 2))
        m5_B = torch.mean(torch.mean((f5_real_B - f5_rec_B) ** 2))

        perceptual_loss_A = m4_A + m5_A
        perceptual_loss_B = m4_B + m5_B
    elif perceptual_layers == '2-4':

        m2_A = torch.mean(torch.mean((f2_real_A - f2_rec_A) ** 2))
        m4_A = torch.mean(torch.mean((f4_real_A - f4_rec_A) ** 2))

        m2_B = torch.mean(torch.mean((f2_real_B - f2_rec_B) ** 2))
        m4_B = torch.mean(torch.mean((f4_real_B - f4_rec_B) ** 2))

        perceptual_loss_A = m2_A + m4_A
        perceptual_loss_B = m2_B + m4_B

    return perceptual_loss_A, perceptual_loss_B


# EXTRACT THE POOLING LAYERS FROM THE PRETRAINED VGG-16
def vgg_16_pretrained(image, model):
    wrapped_model = Inspect(model,
                            layer=['features.4', 'features.9', 'features.16', 'features.23', 'features.30'])
    _, [p1, p2, p3, p4, p5] = wrapped_model(image)

    return p1, p2, p3, p4, p5

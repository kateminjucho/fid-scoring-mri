'''
Created by Minjuc on Mar.13.2023
'''
import torch
from PIL import Image
import timm
from utils import img_to_array, norm_dcm_array, preprocess
import glob
import numpy as np
from distances import EuclideanDistance, CosineSimilarity
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import cv2
from custom_model import CustomNet
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.transforms as transforms


def save_image_pairs(standard_path, swift_path, swift_recon_low_path, swift_recon_medium_path):
    for n in tqdm(range(len(standard_path))):
        out_img = Image.new('RGB', (1024, 1024))
        out_img.paste(Image.fromarray(norm_dcm_array(img_to_array(standard_path[n]))), (0, 0))
        out_img.paste(Image.fromarray(norm_dcm_array(img_to_array(swift_path[n]))), (512, 0))
        out_img.paste(Image.fromarray(norm_dcm_array(img_to_array(swift_recon_low_path[n]))), (0, 512))
        out_img.paste(Image.fromarray(norm_dcm_array(img_to_array(swift_recon_medium_path[n]))), (512, 512))
        out_img.save('./%03d.png' % n)


def feature_computation(m, distance, standard_path, swift_path, swift_recon_low_path, swift_recon_medium_path):
    sim_matrix = np.zeros([4, 4])  # * len(standard_path)
    sim_list = []

    for n in tqdm(range(85)):
        standard_img = Image.fromarray(norm_dcm_array(img_to_array(standard_path[n]))).convert("RGB")
        swift_img = Image.fromarray(norm_dcm_array(img_to_array(swift_path[n]))).convert("RGB")
        swift_recon_low_img = Image.fromarray(norm_dcm_array(img_to_array(swift_recon_low_path[n]))).convert("RGB")
        swift_recon_medium_img = Image.fromarray(norm_dcm_array(img_to_array(swift_recon_medium_path[n]))).convert(
            "RGB")

        file_path_list = [standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img]
        img_list: list = [preprocess(fp) for fp in file_path_list]

        feature_list: list = [m(img.unsqueeze(0)).squeeze(0).detach().numpy() for img in img_list]
        for i in range(len(feature_list)):
            for j in range(i, len(feature_list)):
                distance_tmp = distance(feature_list[i], feature_list[j])
                sim_matrix[i, j] += distance_tmp
                if i != j:
                    sim_matrix[j, i] += distance_tmp
                if i == 0 and j == 2:
                    sim_list.append(distance_tmp)

    sim_matrix = sim_matrix / len(standard_path)
    fig = plt.figure()
    plt.imshow(sim_matrix)
    fig.savefig('./eud_sim_matrix_512.png')


def grad_cam(m, standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img):
    file_path_list = [standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img]
    img_list: list = [preprocess(fp) for fp in file_path_list]
    m_before = torch.nn.Sequential(*list(m.children())[:10])
    m_after = torch.nn.Sequential(*list(m.children())[10:])
    with torch.no_grad():
        img1 = img_list[0].unsqueeze(0)
        img2 = img_list[2].unsqueeze(0)
        feat1 = m_before(img1)
        feat2 = m_before(img2)

    feat2.requires_grad = True

    out1 = m_after(feat1)
    out2 = m_after(feat2)
    # diff = torch.mean(d[0, d_idx[0, -10:]])

    diff = F.mse_loss(out1, out2, reduction='sum')
    # diff = F.cosine_similarity(out1, out2)

    diff.backward(retain_graph=True)

    grads = feat2.grad

    weights = torch.mean(grads, axis=(2, 3), keepdim=True)
    # weights = grads

    cam = torch.abs((weights * feat2).sum(axis=1, keepdim=True)).detach().numpy()[0, 0]
    # cam = weights.sum(axis=1, keepdim=True).detach().numpy()[0, 0]
    # Normalize the gradients
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1E-9)

    red_grads = (255 * cam).astype(np.uint8)
    heatmap = Image.fromarray(cv2.applyColorMap(red_grads, cv2.COLORMAP_JET)).resize((512, 512),
                                                                                     resample=Image.Resampling.BICUBIC)
    out_img = Image.new('RGB', (512 * 3, 512))
    out_img.paste(swift_recon_low_img, (0, 0))
    out_img.paste(Image.blend(swift_recon_low_img, heatmap, 0.5), (512, 0))
    out_img.paste(standard_img, (512 * 2, 0))
    return out_img


def main():
    standard_path = sorted(glob.glob("./AM002_20220812_3691639/standard/*/*.dcm"))
    swift_path = sorted(glob.glob("./AM002_20220812_3691639/swift/*/*.dcm"))
    # swift_recon_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon/6_T1 AX FSE_swift/*.dcm"))
    swift_recon_low_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_low/*/*.dcm"))
    swift_recon_medium_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_medium/*/*.dcm"))
    sim_list = []

    m = timm.create_model('inception_v3', pretrained=True, num_classes=0)
    for n in range(50):
        standard_img = Image.fromarray(norm_dcm_array(img_to_array(standard_path[n]))).convert("RGB")
        swift_img = Image.fromarray(norm_dcm_array(img_to_array(swift_path[n]))).convert("RGB")
        swift_recon_low_img = Image.fromarray(norm_dcm_array(img_to_array(swift_recon_low_path[n]))).convert("RGB")
        swift_recon_medium_img = Image.fromarray(norm_dcm_array(img_to_array(swift_recon_medium_path[n]))).convert(
            "RGB")

        grad_cam(m, standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img).save(
            './mse_17_%02d-feat.png' % (n))


if __name__ == '__main__':
    main()

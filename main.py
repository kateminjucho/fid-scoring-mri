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


def grad_cam(m, standard_img, recon_img, level: int = 10):
    file_path_list = [standard_img, recon_img]
    # Preprocess 해서 list 형태로 만들어 둠.
    img_list: list = [preprocess(fp) for fp in file_path_list]

    # 모델을 두 부분으로 나눔 (m_before, m_after). 숫자는 layer index
    m_before = torch.nn.Sequential(*list(m.children())[:level])
    m_after = torch.nn.Sequential(*list(m.children())[level:])

    # m_before을 forwarding 해서 feature 뽑음. 이 feature에 대해서 gradCAM 적용
    with torch.no_grad():
        img1 = img_list[0].unsqueeze(0)
        img2 = img_list[1].unsqueeze(0)
        feat1 = m_before(img1)
        feat2 = m_before(img2)

    # feature 2에 대해서는 gradient 계산해야 함.
    feat2.requires_grad = True

    # m_after을 forwarding 해서 최종 feature 뽑음. 이 feature로 두 이미지의 차이를 구함.
    out1 = m_after(feat1)
    out2 = m_after(feat2)
    # diff = torch.mean(d[0, d_idx[0, -10:]])

    # feature 차이 구함.
    # indices = torch.square(out1 - out2).argsort(dim=1)[0][-10:]
    diff = F.mse_loss(out1[:, :], out2[:, :], reduction='sum')

    # backpropagation 해서 gradient 계산하기.
    diff.backward(retain_graph=True)

    # feat2에 저장되어 있는 gradient 가져오기
    grads = feat2.grad

    # 각 channel 방향의 feature에 대해서 중요도 계산 (width, height 평균)
    weights = torch.mean(grads, axis=(2, 3), keepdim=True)

    # Weight * feature 해서 gradcam 계산
    cam = torch.abs((weights * feat2).mean(axis=1, keepdim=True)).detach().numpy()[0, 0]
    # Normalize the gradcam
    cam_norm = 1 - ((cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1E-9))

    # 결과 저장
    red_grads = (255 * cam_norm).astype(np.uint8)
    heatmap = Image.fromarray(cv2.applyColorMap(red_grads, cv2.COLORMAP_JET)).resize((512, 512),
                                                                                     resample=Image.Resampling.BICUBIC)
    out_img = Image.new('RGB', (512 * 3, 512))
    out_img.paste(recon_img, (0, 0))
    out_img.paste(Image.blend(recon_img, heatmap, 0.5), (512, 0))
    out_img.paste(standard_img, (512 * 2, 0))
    return out_img, cam_norm


def main():
    # standard_path = sorted(glob.glob("./AM002_20220812_3691639/standard/*/*.dcm"))
    # swift_path = sorted(glob.glob("./AM002_20220812_3691639/swift/*/*.dcm"))
    # swift_recon_low_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_low/*/*.dcm"))
    # swift_recon_medium_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_medium/*/*.dcm"))

    standard_path = sorted(glob.glob("/Volumes/AIRS_CR6/cmc_s_knee/standard/115/*/*.dcm"))
    recon_path = sorted(glob.glob("/Volumes/AIRS_CR6/cmc_s_knee/recon_M/115/*/*.dcm"))

    m = timm.create_model('inception_v3', pretrained=True, num_classes=0)
    for n in range(20, 110):
        grads_gathered = np.zeros([512, 512]).astype(np.float32)
        for level in range(0, 19):
            standard_img = Image.fromarray(norm_dcm_array(img_to_array(standard_path[n]))).convert("RGB")
            recon_img = Image.fromarray(norm_dcm_array(img_to_array(recon_path[n]))).convert("RGB")
            # swift_recon_low_img = Image.fromarray(norm_dcm_array(img_to_array(swift_recon_low_path[n]))).convert("RGB")
            # swift_recon_medium_img = Image.fromarray(norm_dcm_array(img_to_array(swift_recon_medium_path[n]))).convert(
            #     "RGB")

            grad_cam_img, grads = grad_cam(m, standard_img, recon_img, level)
            grads = cv2.resize(grads, (512, 512), interpolation=cv2.INTER_CUBIC)
            grads_gathered += grads

        grads_gathered_norm = 255 * (grads_gathered - grads_gathered.min()) / (grads_gathered.max() - grads_gathered.min())
        heatmap = Image.fromarray(cv2.applyColorMap(grads_gathered_norm.astype(np.uint8), cv2.COLORMAP_JET))

        recon_img = recon_img.resize((512, 512), resample=Image.Resampling.BICUBIC)
        diff = (np.array(standard_img).astype(np.float32) - np.array(recon_img).astype(np.float32) + 255) / 2
        diff_img = Image.fromarray(cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_JET))
        final_img = Image.new('RGB', (512 * 4, 512))
        final_img.paste(recon_img, (0, 0))
        final_img.paste(Image.blend(recon_img, heatmap, 0.5), (512, 0))
        final_img.paste(standard_img, (512 * 2, 0))
        final_img.paste(diff_img, (512 * 3, 0))
        final_img.save('./result/cmc_knee/final_%02d.png' % n)


if __name__ == '__main__':
    main()

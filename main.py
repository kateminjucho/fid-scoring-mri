'''
Created by Minjuc on Mar.13.2023
'''
import torch
from PIL import Image
import timm
from utils import img_to_array, norm_dcm_array, preprocess
import glob
import numpy as np
from distances import EuclideanDistance
import matplotlib.pyplot as plt


def main():
    standard_path = sorted(glob.glob("./AM002_20220812_3691639/standard/5_T1 AX FSE/*.dcm"))
    swift_path = sorted(glob.glob("./AM002_20220812_3691639/swift/6_T1 AX FSE_swift/*.dcm"))
    # swift_recon_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon/6_T1 AX FSE_swift/*.dcm"))
    swift_recon_low_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_low/6_T1 AX FSE_swift/*.dcm"))
    swift_recon_medium_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_medium/6_T1 AX FSE_swift/*.dcm"))
    standard_img = Image.fromarray(norm_dcm_array(img_to_array(standard_path[0]))).convert("RGB")
    swift_img = Image.fromarray(norm_dcm_array(img_to_array(swift_path[0]))).convert("RGB")
    swift_recon_low_img = Image.fromarray(norm_dcm_array(img_to_array(swift_recon_low_path[0]))).convert("RGB")
    swift_recon_medium_img = Image.fromarray(norm_dcm_array(img_to_array(swift_recon_medium_path[0]))).convert("RGB")

    file_path_list = [standard_img, swift_img, swift_recon_low_img, swift_recon_medium_img]
    img_list: list = [preprocess(fp) for fp in file_path_list]

    distance = EuclideanDistance()

    m = timm.create_model('inception_v3', pretrained=True, num_classes=0)
    feature_list: list = [m(img.unsqueeze(0)).squeeze(0).detach().numpy() for img in img_list]
    sim_matrix = np.zeros([len(feature_list), len(feature_list)])
    for i in range(len(feature_list)):
        for j in range(i + 1, len(feature_list)):
            distance_tmp = distance(feature_list[i], feature_list[j])
            sim_matrix[i, j] = distance_tmp
            sim_matrix[j, i] = distance_tmp

    fig = plt.figure()
    plt.imshow(sim_matrix)
    fig.savefig('./sim_matrix.png')


if __name__ == '__main__':
    main()

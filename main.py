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


def main():
    standard_path = sorted(glob.glob("./AM002_20220812_3691639/standard/*/*.dcm"))
    swift_path = sorted(glob.glob("./AM002_20220812_3691639/swift/*/*.dcm"))
    # swift_recon_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon/6_T1 AX FSE_swift/*.dcm"))
    swift_recon_low_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_low/*/*.dcm"))
    swift_recon_medium_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_medium/*/*.dcm"))
    sim_matrix = np.zeros([4, 4])  # * len(standard_path)

    distance = CosineSimilarity()
    m = timm.create_model('inception_v3', pretrained=True, num_classes=0)

    for n in tqdm(range(len(standard_path))):
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

    sim_matrix = sim_matrix / len(standard_path)
    fig = plt.figure()
    plt.imshow(sim_matrix)
    fig.savefig('./cosine_sim_matrix.png')


if __name__ == '__main__':
    main()

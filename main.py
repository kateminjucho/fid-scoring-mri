'''
Created by Minjuc on Mar.13.2023
'''
import torch
from torchvision import transforms
from PIL import Image
import timm
from utils import img_to_array, norm_dcm_array
import glob


def load_png(file_path: str) -> Image.Image:
    return Image.open(file_path)


def preprocess(img: Image.Image) -> Image.Image:
    pipeline = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return pipeline(img)


def main():
    standard_path = sorted(glob.glob("./AM002_20220812_3691639/standard/5_T1 AX FSE/*.dcm"))
    recon_path = sorted(glob.glob("./AM002_20220812_3691639/swift_recon_low/6_T1 AX FSE_swift/*.dcm"))
    standard_img = Image.fromarray(norm_dcm_array(img_to_array(standard_path[0]))).convert("RGB")
    recon_img = Image.fromarray(norm_dcm_array(img_to_array(recon_path[0]))).convert("RGB")
    out_img = Image.new('RGB', (1024, 512))
    out_img.paste(standard_img, (0, 0))
    out_img.paste(recon_img, (512, 0))
    out_img.save('./result.png')

    file_path_list = [standard_img, recon_img]
    img_list: list = [preprocess(fp) for fp in file_path_list]

    m = timm.create_model('inception_v3', pretrained=True, num_classes=0)
    out1 = m(img_list[0].unsqueeze(0))
    out2 = m(img_list[1].unsqueeze(0))
    print('Hello world')


if __name__ == '__main__':
    main()

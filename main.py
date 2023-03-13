'''
Created by Minjuc on Mar.13.2023
'''
import torch
from torchvision import transforms
from PIL import Image
import timm


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
    file_path_list = ['img1.png', 'img2.png']
    img_list: list = [preprocess(load_png(fp)) for fp in file_path_list]

    m = timm.create_model('inception_v3', pretrained=True, num_classes=0)
    out1 = m(img_list[0].unsqueeze(0))
    out2 = m(img_list[1].unsqueeze(0))
    print('Hello world')


if __name__ == '__main__':
    main()

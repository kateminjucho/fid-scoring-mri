from pydicom import dcmread
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


def img_to_array(dcm_path: str):
    try:
        img = dcmread(dcm_path).pixel_array
    except FileNotFoundError:
        print('File does not exists: %s' % dcm_path)
        return None
    return img


def norm_dcm_array(dcm_array: np.ndarray, low: int = None, high: int = None) -> np.ndarray:
    if low is None:
        low = np.min(dcm_array)
    if high is None:
        high = np.max(dcm_array)

    assert low < high

    return (np.clip((dcm_array - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def load_png(file_path: str) -> Image.Image:
    return Image.open(file_path)


class Crop:
    def __init__(self):
        self.left = 128
        self.top = 256
        self.width = 128
        self.height = 128

    def __call__(self, img):
        return F.crop(img, self.top, self.left, self.height, self.width)


def preprocess(img: Image.Image) -> Image.Image:
    pipeline = transforms.Compose([
        transforms.Resize(512),
        # Crop(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return pipeline(img)

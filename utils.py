from pydicom import dcmread
import numpy as np

def img_to_array(dcm_path):
    img = dcmread(dcm_path).pixel_array
    max = np.max(img)
    min = np.min(img)
    img_norm = ((img - min) / (max - min) * 255).astype(np.uint8)
    return img_norm
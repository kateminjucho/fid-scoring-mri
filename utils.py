from pydicom import dcmread
import numpy as np


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

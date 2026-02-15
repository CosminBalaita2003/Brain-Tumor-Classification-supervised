import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog

def extract_features(img_path: str) -> np.ndarray:

    img = imread(img_path)
    if img.ndim == 3:
        img = rgb2gray(img)

    img = resize(img, (128, 128), anti_aliasing=True).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    hog_vec = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return hog_vec.astype(np.float32)

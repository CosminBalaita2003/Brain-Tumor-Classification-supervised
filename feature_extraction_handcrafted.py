import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from skimage.feature import hog


def extract_features(img_path: str) -> np.ndarray:
    img = imread(img_path)
    if img.ndim == 3 and img.shape[2] == 3:
        img = rgb2gray(img)

    img = resize(img, (128, 128), anti_aliasing=True).astype(np.float32)

    img = np.clip(img, 0.0, 1.0)

    features = []
    pixels = img.ravel()

    # 1) stats
    features += [
        float(pixels.mean()),
        float(pixels.std()),
        float(skew(pixels)),
        float(kurtosis(pixels)),
    ]

    histo, _ = np.histogram(pixels, bins=32, range=(0.0, 1.0), density=True)
    features += histo.astype(np.float32).tolist()

    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)


    lbp = local_binary_pattern(img_u8, P=8, R=1, method="uniform")
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    features += lbp_hist.astype(np.float32).tolist()

    levels = 32
    quant = (img_u8 * (levels - 1) // 255).astype(np.uint8)

    glcm = graycomatrix(
        quant,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=levels,
        symmetric=True,
        normed=True
    )

    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        features += graycoprops(glcm, prop).ravel().astype(np.float32).tolist()

    print(f"Extracted {len(features)} features from {img_path}")

    return np.array(features, dtype=np.float32)





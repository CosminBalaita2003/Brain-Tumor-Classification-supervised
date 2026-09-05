import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_float32
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew, kurtosis


def extract_features(img_path: str) -> np.ndarray:
    img = imread(img_path)
    if img.ndim == 3:
        # Explicit conversion supports RGB/RGBA and avoids version-specific
        # matrix-multiplication warnings observed with skimage.rgb2gray.
        rgb = img_as_float32(img[..., :3])
        img = 0.2125 * rgb[..., 0] + 0.7154 * rgb[..., 1] + 0.0721 * rgb[..., 2]
    else:
        img = img_as_float32(img)

    img = resize(img, (128, 128), anti_aliasing=True).astype(np.float32)

    img = np.clip(img, 0.0, 1.0)

    features = []
    pixels = img.ravel()

    # 1) stats
    features += [
        float(pixels.mean()),
        float(pixels.std()),
        float(np.nan_to_num(skew(pixels))),
        float(np.nan_to_num(kurtosis(pixels))),
    ]

    histo, _ = np.histogram(pixels, bins=32, range=(0.0, 1.0), density=True)
    features += histo.astype(np.float32).tolist()

    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)


    lbp = local_binary_pattern(img_u8, P=8, R=1, method="uniform")
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    features += lbp_hist.astype(np.float32).tolist()

    levels = 32
    # Promote before multiplication; uint8 arithmetic overflowed here and
    # collapsed most GLCM intensities into the lowest bin.
    quant = (img_u8.astype(np.uint16) * (levels - 1) // 255).astype(np.uint8)

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

    return np.array(features, dtype=np.float32)



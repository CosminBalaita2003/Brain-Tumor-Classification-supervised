"""Extract compact multi-scale statistics from a 2D wavelet decomposition."""

import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_float32

IMAGE_SIZE = (128, 128)
WAVELET = "db2"
LEVELS = 3
HISTOGRAM_BINS = 32


def _band_statistics(band: np.ndarray) -> list[float]:
    """Summarize one wavelet sub-band without depending on its dimensions."""
    values = np.asarray(band, dtype=np.float32).ravel()
    absolute = np.abs(values)
    probabilities, _ = np.histogram(absolute, bins=HISTOGRAM_BINS)
    probabilities = probabilities.astype(np.float64)
    probabilities /= max(probabilities.sum(), 1.0)
    nonzero = probabilities[probabilities > 0]

    return [
        float(values.mean()),
        float(values.std()),
        float(np.mean(values * values)),
        float(absolute.mean()),
        float(-np.sum(nonzero * np.log2(nonzero))),
        float(np.nan_to_num(skew(values))),
        float(np.nan_to_num(kurtosis(values))),
    ]


def extract_features(img_path: str) -> np.ndarray:
    """Return statistics for approximation and detail bands at three scales."""
    img = imread(img_path)
    if img.ndim == 3:
        rgb = img_as_float32(img[..., :3])
        img = 0.2125 * rgb[..., 0] + 0.7154 * rgb[..., 1] + 0.0721 * rgb[..., 2]
    else:
        img = img_as_float32(img)

    img = resize(img, IMAGE_SIZE, anti_aliasing=True).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    coefficients = pywt.wavedec2(img, wavelet=WAVELET, level=LEVELS, mode="symmetric")

    bands = [coefficients[0]]
    for horizontal, vertical, diagonal in coefficients[1:]:
        bands.extend((horizontal, vertical, diagonal))

    features = [value for band in bands for value in _band_statistics(band)]
    return np.asarray(features, dtype=np.float32)

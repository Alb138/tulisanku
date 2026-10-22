"""Two baselines that do not generate anything - every real model must beat them.

MeanGlyph    always returns the average of all training fonts for each character:
             no personalization at all, the "dumbest reasonable guess".
NearestFont  finds the training font whose letters look most like the k reference
             letters and returns that font's other letters. Matching is done on
             blurred 32x32 copies: faster, and tolerant of small shifts.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def shrink(images: np.ndarray, size: int = 32, blur: float = 1.0) -> np.ndarray:
    """[..., H, W] uint8 -> [..., size*size] float32 in 0..1, blurred then averaged down."""
    x = np.asarray(images, np.float32) / 255
    x = gaussian_filter(x, sigma=(0,) * (x.ndim - 2) + (blur, blur))
    f = x.shape[-1] // size
    x = x.reshape(*x.shape[:-2], size, f, size, f).mean(axis=(-3, -1))
    return x.reshape(*x.shape[:-2], size * size)


class MeanGlyph:
    name = "Mean glyph"

    def __init__(self, font_images):
        total = np.zeros(font_images.shape[1:], np.float64)
        for start in range(0, len(font_images), 64):  # in chunks: the fonts stay on disk
            total += np.asarray(font_images[start:start + 64]).sum(axis=0, dtype=np.float64)
        self.mean = (total / len(font_images) / 255).astype(np.float32)

    def predict(self, ref_images, ref_idx, target_idx):
        return self.mean[target_idx]


class NearestFont:
    name = "Nearest font"

    def __init__(self, font_images, size: int = 32, blur: float = 1.0):
        self.fonts, self.size, self.blur = font_images, size, blur
        self.library = np.concatenate([shrink(font_images[s:s + 64], size, blur).astype(np.float16)
                                       for s in range(0, len(font_images), 64)])  # [n_fonts, 62, size*size]
        self.last_choice = None

    def predict(self, ref_images, ref_idx, target_idx):
        refs = shrink(ref_images, self.size, self.blur)  # [k, size*size]
        distance = np.zeros(len(self.library), np.float32)
        for j, char in enumerate(ref_idx):
            distance += np.abs(self.library[:, char].astype(np.float32) - refs[j]).mean(axis=1)
        self.last_choice = int(np.argmin(distance))
        return np.asarray(self.fonts[self.last_choice, target_idx], np.float32) / 255

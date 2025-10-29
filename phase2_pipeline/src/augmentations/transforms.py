"""
Image Augmentation Transforms for Test-Time Augmentation
Borrowed from albumentations and adapted for TTA
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class BaseTransform:
    """Base class for all transforms"""

    def __init__(self, p: float = 1.0):
        """
        Initialize transform

        Args:
            p: Probability of applying transform
        """
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply transform with probability p"""
        if np.random.random() < self.p:
            return self.apply(image)
        return image

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the transform (to be implemented by subclasses)"""
        raise NotImplementedError


class GaussianBlur(BaseTransform):
    """
    Apply Gaussian Blur
    Borrowed from albumentations/augmentations/blur/transforms.py
    """

    def __init__(
        self,
        blur_limit: Tuple[int, int] = (3, 7),
        sigma_limit: float = 0,
        p: float = 1.0
    ):
        """
        Initialize Gaussian Blur

        Args:
            blur_limit: Kernel size range (must be odd)
            sigma_limit: Sigma range, 0 for auto-calculation
            p: Probability of applying
        """
        super().__init__(p)
        self.blur_limit = blur_limit
        self.sigma_limit = sigma_limit

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur"""
        # Random kernel size
        ksize = np.random.randint(self.blur_limit[0], self.blur_limit[1] + 1)
        # Ensure odd
        if ksize % 2 == 0:
            ksize += 1

        # Auto-calculate sigma if not specified
        if self.sigma_limit == 0:
            sigma = ksize / 3.0
        else:
            sigma = np.random.uniform(0.1, self.sigma_limit)

        # Apply blur
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        return blurred


class GaussianNoise(BaseTransform):
    """
    Add Gaussian Noise
    Borrowed from albumentations/augmentations/transforms.py
    """

    def __init__(
        self,
        var_limit: Tuple[float, float] = (10.0, 50.0),
        mean: float = 0,
        per_channel: bool = True,
        p: float = 1.0
    ):
        """
        Initialize Gaussian Noise

        Args:
            var_limit: Variance range for noise
            mean: Mean of the noise
            per_channel: Apply different noise per channel
            p: Probability of applying
        """
        super().__init__(p)
        self.var_limit = var_limit
        self.mean = mean
        self.per_channel = per_channel

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        var = np.random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5

        if self.per_channel and len(image.shape) == 3:
            # Different noise per channel
            noise = np.random.normal(self.mean, sigma, image.shape)
        else:
            # Same noise for all channels
            noise_shape = image.shape[:2] if len(image.shape) == 3 else image.shape
            noise = np.random.normal(self.mean, sigma, noise_shape)
            if len(image.shape) == 3:
                noise = np.repeat(noise[:, :, np.newaxis], image.shape[2], axis=2)

        # Add noise and clip
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return noisy


class RandomBrightnessContrast(BaseTransform):
    """
    Adjust brightness and contrast
    Borrowed from albumentations/augmentations/transforms.py
    """

    def __init__(
        self,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        brightness_by_max: bool = True,
        p: float = 1.0
    ):
        """
        Initialize brightness/contrast adjustment

        Args:
            brightness_limit: Brightness change limit (-limit, +limit)
            contrast_limit: Contrast change limit (-limit, +limit)
            brightness_by_max: Scale by max value (255) or mean
            p: Probability of applying
        """
        super().__init__(p)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_by_max = brightness_by_max

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Adjust brightness and contrast"""
        # Random factors
        brightness_factor = np.random.uniform(
            -self.brightness_limit,
            self.brightness_limit
        )
        contrast_factor = np.random.uniform(
            1 - self.contrast_limit,
            1 + self.contrast_limit
        )

        # Brightness adjustment
        if self.brightness_by_max:
            brightness_delta = brightness_factor * 255
        else:
            brightness_delta = brightness_factor * np.mean(image)

        # Apply adjustments
        img_float = image.astype(np.float32)
        img_float = img_float * contrast_factor + brightness_delta
        adjusted = np.clip(img_float, 0, 255).astype(np.uint8)

        return adjusted


class RandomScale(BaseTransform):
    """
    Random scale (zoom in/out)
    """

    def __init__(
        self,
        scale_limit: float = 0.1,
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1.0
    ):
        """
        Initialize random scale

        Args:
            scale_limit: Scale range (1-limit, 1+limit)
            interpolation: OpenCV interpolation method
            p: Probability of applying
        """
        super().__init__(p)
        self.scale_limit = scale_limit
        self.interpolation = interpolation

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply random scale"""
        h, w = image.shape[:2]

        # Random scale factor
        scale = np.random.uniform(1 - self.scale_limit, 1 + self.scale_limit)

        # New dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)

        # Crop or pad to original size
        if scale > 1:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            result = resized[start_h:start_h + h, start_w:start_w + w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            result = np.zeros_like(image)
            result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        return result


class JPEGCompression(BaseTransform):
    """
    JPEG compression artifacts
    Borrowed from albumentations/augmentations/transforms.py
    """

    def __init__(
        self,
        quality_lower: int = 40,
        quality_upper: int = 100,
        p: float = 1.0
    ):
        """
        Initialize JPEG compression

        Args:
            quality_lower: Lower bound of quality
            quality_upper: Upper bound of quality
            p: Probability of applying
        """
        super().__init__(p)
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply JPEG compression"""
        # Random quality
        quality = np.random.randint(self.quality_lower, self.quality_upper + 1)

        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)

        # Decode back
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)

        return compressed


class TTACompose:
    """
    Compose multiple transforms for TTA
    Similar to ttach/base.py
    """

    def __init__(self, transforms: List[BaseTransform]):
        """
        Initialize TTA composition

        Args:
            transforms: List of transforms to apply
        """
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply all transforms and return augmented images

        Args:
            image: Input image

        Returns:
            List of augmented images (including original)
        """
        augmented = [image.copy()]  # Include original

        for transform in self.transforms:
            aug_img = transform(image.copy())
            augmented.append(aug_img)

        return augmented


def get_tta_transforms(config: dict = None) -> TTACompose:
    """
    Get default TTA transforms based on config

    Args:
        config: Configuration dictionary

    Returns:
        TTACompose object with transforms
    """
    if config is None:
        config = {}

    transforms = []

    # Gaussian Blur
    if config.get('gaussian_blur', {}).get('enabled', True):
        blur_config = config.get('gaussian_blur', {})
        transforms.append(GaussianBlur(
            blur_limit=blur_config.get('blur_limit', (3, 7)),
            sigma_limit=blur_config.get('sigma_limit', 0)
        ))

    # Gaussian Noise
    if config.get('gaussian_noise', {}).get('enabled', True):
        noise_config = config.get('gaussian_noise', {})
        transforms.append(GaussianNoise(
            var_limit=noise_config.get('var_limit', (10.0, 50.0)),
            mean=noise_config.get('mean', 0),
            per_channel=noise_config.get('per_channel', True)
        ))

    # Brightness/Contrast
    if config.get('brightness_contrast', {}).get('enabled', True):
        bc_config = config.get('brightness_contrast', {})
        transforms.append(RandomBrightnessContrast(
            brightness_limit=bc_config.get('brightness_limit', 0.2),
            contrast_limit=bc_config.get('contrast_limit', 0.2),
            brightness_by_max=bc_config.get('brightness_by_max', True)
        ))

    # Scale
    if config.get('scale', {}).get('enabled', True):
        scale_config = config.get('scale', {})
        transforms.append(RandomScale(
            scale_limit=scale_config.get('scale_limit', 0.1),
            interpolation=scale_config.get('interpolation', 1)
        ))

    # JPEG Compression
    if config.get('jpeg_compression', {}).get('enabled', True):
        jpeg_config = config.get('jpeg_compression', {})
        transforms.append(JPEGCompression(
            quality_lower=jpeg_config.get('quality_lower', 40),
            quality_upper=jpeg_config.get('quality_upper', 100)
        ))

    logger.info(f"Created TTA with {len(transforms)} augmentations")

    return TTACompose(transforms)


if __name__ == "__main__":
    # Test augmentations
    import sys
    sys.path.append('/ssd_4TB/divake/temporal_uncertainty')

    # Create dummy image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128

    # Test individual transforms
    blur = GaussianBlur()
    noise = GaussianNoise()
    brightness = RandomBrightnessContrast()
    scale = RandomScale()
    jpeg = JPEGCompression()

    print("Testing transforms:")
    print(f"  Original shape: {img.shape}, mean: {img.mean():.1f}")

    blurred = blur(img)
    print(f"  Blurred mean: {blurred.mean():.1f}")

    noisy = noise(img)
    print(f"  Noisy std: {noisy.std():.1f}")

    bright = brightness(img)
    print(f"  Brightness adjusted mean: {bright.mean():.1f}")

    scaled = scale(img)
    print(f"  Scaled shape: {scaled.shape}")

    compressed = jpeg(img)
    print(f"  JPEG compressed mean: {compressed.mean():.1f}")

    # Test TTA compose
    print("\nTesting TTA Compose:")
    tta = get_tta_transforms()
    augmented = tta(img)
    print(f"  Generated {len(augmented)} augmented images (including original)")
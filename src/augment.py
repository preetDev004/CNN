from typing import Tuple

import cv2
import numpy as np


def pan(img: np.ndarray) -> np.ndarray:
    """
    Randomly translate an image horizontally and vertically by up to
    10% of its width/height, simulating slight off-center framing.

    Args:
        img: Image as an (H, W, C) array.

    Returns:
        Translated image, same shape as input. Edge pixels are
        replicated to fill the newly exposed border.
    """
    h, w = img.shape[:2]
    tx = np.random.uniform(-0.1, 0.1) * w
    ty = np.random.uniform(-0.1, 0.1) * h
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def zoom(img: np.ndarray) -> np.ndarray:
    """
    Randomly scale an image up by 1.0x-1.3x around its center,
    simulating the car being closer to the road features.

    Args:
        img: Image as an (H, W, C) array.

    Returns:
        Zoomed image, same shape as input.
    """
    h, w = img.shape[:2]
    scale = np.random.uniform(1.0, 1.3)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def rotate(img: np.ndarray) -> np.ndarray:
    """
    Randomly rotate an image by up to 10 degrees around its center,
    simulating a tilted camera mount.

    Args:
        img: Image as an (H, W, C) array.

    Returns:
        Rotated image, same shape as input.
    """
    h, w = img.shape[:2]
    angle = np.random.uniform(-10, 10)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def adjust_brightness(img: np.ndarray) -> np.ndarray:
    """
    Randomly scale an image's brightness by 0.4x-1.3x, simulating
    different lighting/time-of-day conditions.

    Args:
        img: RGB image as an (H, W, 3) uint8 array.

    Returns:
        Brightness-adjusted RGB image, same shape and dtype as input.
    """
    factor = np.random.uniform(0.4, 1.3)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float64)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def flip(img: np.ndarray, steering: float) -> Tuple[np.ndarray, float]:
    """
    Horizontally flip an image and negate its steering angle to match.

    Args:
        img: Image as an (H, W, C) array.
        steering: Steering angle corresponding to img.

    Returns:
        Tuple of (flipped_image, negated_steering).
    """
    img = cv2.flip(img, 1)
    steering = -steering
    return img, steering


def augment_image(img: np.ndarray, steering: float) -> Tuple[np.ndarray, float]:
    """
    Apply a random subset of augmentations to one training sample.

    Each of pan/zoom/rotate/brightness/flip is applied independently
    with 50% probability, so a given image can receive 0 to 5 stacked
    transforms. Only flip changes the steering label.

    Args:
        img: Raw RGB image as an (H, W, 3) uint8 array.
        steering: Steering angle corresponding to img.

    Returns:
        Tuple of (augmented_image, augmented_steering).
    """
    if np.random.rand() < 0.5:
        img = pan(img)
    if np.random.rand() < 0.5:
        img = zoom(img)
    if np.random.rand() < 0.5:
        img = rotate(img)
    if np.random.rand() < 0.5:
        img = adjust_brightness(img)
    if np.random.rand() < 0.5:
        img, steering = flip(img, steering)
    return img, steering

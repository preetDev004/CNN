import cv2
import numpy as np

CROP_TOP: int = 60
CROP_BOTTOM: int = 135
TARGET_WIDTH: int = 200
TARGET_HEIGHT: int = 66


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Preprocess a single raw camera frame for the model.

    Applies, in order: crop to the road area, convert to YUV color
    space, Gaussian blur, resize to the Nvidia PilotNet input size,
    and normalize to [0, 1]. The order matches TestSimulation.py's
    preProcessing() exactly, changing it here without changing it
    there (or vice versa) will silently break train/inference
    consistency.

    Args:
        img: Raw RGB image as an (H, W, 3) uint8 array, as read from
            the simulator's camera output or a recorded JPEG frame.

    Returns:
        Preprocessed image as a (66, 200, 3) array with values in
        [0, 1], ready to feed to the model.
    """
    img = img[CROP_TOP:CROP_BOTTOM, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
    img = img / 255
    return img
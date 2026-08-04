from typing import Iterator, Tuple

import numpy as np
import matplotlib.image as mpimg

from src.data_preprocessing import preprocess
from src.augment import augment_image


def batch_generator(
    image_paths: np.ndarray,
    steering_angles: np.ndarray,
    batch_size: int,
    is_training: bool,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Infinitely yield randomly-sampled, preprocessed batches.

    Samples are drawn with replacement, so this can run indefinitely
    regardless of how many steps_per_epoch is set to. Training batches
    get augmentation applied on top of preprocessing; validation
    batches only get preprocessing, matching the "augment training
    data only" rule.

    Args:
        image_paths: Array of file paths to center-camera images.
        steering_angles: Array of steering angles, same length and
            index alignment as image_paths.
        batch_size: Number of samples to yield per batch.
        is_training: If True, apply augment_image() before
            preprocessing. If False, preprocess only.

    Yields:
        Tuple of (images, steerings):
            images: float array of shape (batch_size, 66, 200, 3).
            steerings: float array of shape (batch_size,).
    """
    n = len(image_paths)
    while True:
        batch_imgs = []
        batch_steerings = []
        for _ in range(batch_size):
            idx = np.random.randint(0, n)
            img = mpimg.imread(image_paths[idx])
            steering = steering_angles[idx]

            if is_training:
                img, steering = augment_image(img, steering)

            img = preprocess(img)
            batch_imgs.append(img)
            batch_steerings.append(steering)

        yield np.asarray(batch_imgs), np.asarray(batch_steerings)
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]


def load_log(data_dir: str) -> pd.DataFrame:
    """
    Load driving_log.csv into a DataFrame and normalize center-camera
    filenames to just the basename.

    Args:
        data_dir: Directory containing driving_log.csv and an IMG/
            subfolder with the recorded frames.

    Returns:
        DataFrame with columns COLUMNS, where "center" holds just the
        image filename (no directory component).
    """
    csv_path = os.path.join(data_dir, "driving_log.csv")
    df = pd.read_csv(csv_path, names=COLUMNS)
    df["center"] = df["center"].apply(lambda p: p.strip().replace("\\", "/").split("/")[-1])
    return df


def plot_steering_histogram(df: pd.DataFrame, num_bins: int, out_path: str, title: str) -> None:
    """
    Plot and save a histogram of the steering angle distribution.

    Args:
        df: DataFrame with a "steering" column.
        num_bins: Number of histogram bins.
        out_path: File path to save the PNG to.
        title: Plot title.

    Returns:
        None. Writes a PNG file to out_path as a side effect.
    """
    hist, bins = np.histogram(df["steering"], num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5
    plt.figure()
    plt.bar(center, hist, width=0.05)
    plt.title(title)
    plt.xlabel("steering angle")
    plt.ylabel("count")
    plt.savefig(out_path)
    plt.close()


def balance_data(df: pd.DataFrame, num_bins: int = 25, samples_per_bin: int = 400) -> pd.DataFrame:
    """
    Flatten the steering-angle distribution by capping each histogram
    bin at samples_per_bin, randomly dropping the excess rows.

    Straight-line driving dominates raw simulator data, so without
    balancing the model over-learns "steer straight" and struggles on
    corners.

    Args:
        df: DataFrame with a "steering" column.
        num_bins: Number of histogram bins to cap independently.
        samples_per_bin: Max rows to keep per bin. If this is >= the
            largest bin's count, balancing has no effect, a warning is
            logged in that case.

    Returns:
        A new DataFrame with excess rows in over-represented bins
        removed and the index reset.
    """
    hist, bins = np.histogram(df["steering"], num_bins)
    logger.info(
        "Steering bin counts (max=%d, median=%d): %s",
        hist.max(), int(np.median(hist)), hist.tolist(),
    )
    if samples_per_bin >= hist.max():
        logger.warning(
            "samples_per_bin=%d is >= the largest bin (%d), so balancing "
            "will remove nothing. Pass a lower --samples-per-bin (try "
            "something near the median above).",
            samples_per_bin, hist.max(),
        )

    remove_idx = []
    for i in range(num_bins):
        bin_rows = df[(df["steering"] >= bins[i]) & (df["steering"] <= bins[i + 1])].index.tolist()
        bin_rows = np.random.permutation(bin_rows).tolist()
        remove_idx.extend(bin_rows[samples_per_bin:])
    balanced = df.drop(remove_idx).reset_index(drop=True)
    return balanced


def prepare_dataset(
    data_dir: str,
    num_bins: int = 25,
    samples_per_bin: int = 400,
    val_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full data-prep pipeline: load the CSV, plot the steering
    distribution before and after balancing, then split into
    train/validation sets.

    Args:
        data_dir: Directory containing driving_log.csv and IMG/.
        num_bins: Number of histogram bins used for balancing.
        samples_per_bin: Max rows kept per bin during balancing.
        val_size: Fraction of the balanced data held out for
            validation (e.g. 0.2 = 20%).

    Returns:
        Tuple of (X_train, X_val, y_train, y_val):
            X_train: Training image file paths, shape (n_train,).
            X_val: Validation image file paths, shape (n_val,).
            y_train: Training steering angles, shape (n_train,).
            y_val: Validation steering angles, shape (n_val,).
    """
    df = load_log(data_dir)
    plot_steering_histogram(df, num_bins, "training_plots/steering_before.png", "Before balancing")

    df = balance_data(df, num_bins, samples_per_bin)
    plot_steering_histogram(df, num_bins, "training_plots/steering_after.png", "After balancing")

    img_dir = os.path.join(data_dir, "IMG")
    image_paths = df["center"].apply(lambda f: os.path.join(img_dir, f)).to_numpy()
    steerings = df["steering"].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, steerings, test_size=val_size, random_state=42
    )
    logger.info("Train: %d  Val: %d", len(X_train), len(X_val))
    return X_train, X_val, y_train, y_val

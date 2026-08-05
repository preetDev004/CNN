"""
Main training entry point.

Usage:
    uv run train.py

Expects data/driving_log.csv and data/IMG/ to already exist (collected
from the simulator in Training Mode).
"""
import argparse
import logging
import os

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping

from src.utils import prepare_dataset
from src.batch import batch_generator
from src.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training run.

    Returns:
        Namespace with data_dir, epochs, batch_size, steps_per_epoch,
        val_steps, learning_rate, samples_per_bin, and output.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data",
                         help="Directory containing driving_log.csv and IMG/.")
    parser.add_argument("--epochs", type=int, default=10,
                         help="Max training epochs (EarlyStopping may stop sooner).")
    parser.add_argument("--batch-size", type=int, default=100,
                         help="Samples per batch.")
    parser.add_argument("--steps-per-epoch", type=int, default=300,
                         help="Batches drawn per training epoch.")
    parser.add_argument("--val-steps", type=int, default=200,
                         help="Batches drawn per validation epoch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                         help="Adam optimizer learning rate.")
    parser.add_argument("--samples-per-bin", type=int, default=50,
                         help="Max rows kept per steering histogram bin when balancing.")
    parser.add_argument("--output", default="model/model.h5",
                         help="Path to save the trained model to.")
    return parser.parse_args()


def main() -> None:
    """
    Run the full pipeline: load and balance data, train the model with
    checkpointing and early stopping, save the best model, and plot
    the training/validation loss curve.
    """
    args = parse_args()
    os.makedirs("training_plots", exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    X_train, X_val, y_train, y_val = prepare_dataset(
        args.data_dir, samples_per_bin=args.samples_per_bin
    )

    model = build_model(args.learning_rate)
    model.summary()

    checkpoint = ModelCheckpoint(
        args.output,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        batch_generator(X_train, y_train, args.batch_size, is_training=True),
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=batch_generator(X_val, y_val, args.batch_size, is_training=False),
        validation_steps=args.val_steps,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    # EarlyStopping already restored the best weights in memory; make sure
    # the file on disk reflects that too, in case training ran the full
    # epoch count without early stopping ever triggering.
    model.save(args.output)
    logger.info("Best model saved to %s", args.output)

    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.savefig("training_plots/loss_curve.png")
    logger.info("Loss curve saved to training_plots/loss_curve.png")


if __name__ == "__main__":
    main()
import argparse
import os
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from data import get_training_data
from model import create_model, save_model

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TIME_STEPS = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def train(data_path, epochs, batch_size, validation_split, verbose, save_training_plot, model_save_dir):
    X_train, y_train = get_training_data(data_path)
    log.info("Creating model")
    model = create_model(X_train)
    log.info(model.summary())
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose
    )

    save_model(model_save_dir, model)

    if save_training_plot:
        fig = plt.figure(figsize=(9, 6))
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        path_to_save_plot = Path("../results/training/")
        if path_to_save_plot.exists():
            plt.savefig(path_to_save_plot / "training.png")
        else:
            path_to_save_plot.mkdir(exist_ok=True, parents=True)
            plt.savefig(path_to_save_plot / "training.png")


def create_and_parse_args(args=None):
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--data_path', type=Path, default="../data/lstm_data", help='Dir with lstm data')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of elements in batch')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Train/validation split')
    parser.add_argument('--verbose', type=int, default=1, help='Train/validation split')
    parser.add_argument('--save_training_plot', action='store_true', default=True, help='Whether to store trainings plots')
    parser.add_argument('--model_save_dir', type=Path, default="model", help='Dir where model will be dumped')
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = create_and_parse_args()
    train(**vars(args))
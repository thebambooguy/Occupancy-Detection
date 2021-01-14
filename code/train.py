import os
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from data import get_training_data
from model import create_model, save_model

TIME_STEPS = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def train(data_path, epochs=5, batch_size=32, validation_split=0.1, verbose=1, save_training_plot=True, model_save_dir = Path("model")):
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


if __name__ == "__main__":
    # https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # on Unix use 2
    data_path = Path("../data/lstm_data")
    train(data_path)
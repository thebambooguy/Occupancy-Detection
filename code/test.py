import os
import logging
from pathlib import Path

from data import get_test_data
from model import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def test(data_path, model_path):
    X_test, y_test = get_test_data(data_path)
    model = load_model(model_path)
    model.evaluate(X_test, y_test)


if __name__ == "__main__":
    # https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # on Unix use 2
    data_path = Path("../data/lstm_data")
    model_path = Path("model/model.h5")
    test(data_path, model_path)
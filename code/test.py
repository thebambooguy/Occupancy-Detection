import argparse
import os
import logging
from pathlib import Path

from data import get_test_data
from model import load_model

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def test(data_path, model_path):
    X_test, y_test = get_test_data(data_path)
    model = load_model(model_path)
    model.evaluate(X_test, y_test)


def create_and_parse_args(args=None):
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data_path', type=Path, default="../data/lstm_data", help='Dir with lstm data')
    parser.add_argument('--model_path', type=Path, default="model/model.h5", help='Path to dumped model file')
    args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = create_and_parse_args()
    test(**vars(args))
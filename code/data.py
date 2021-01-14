from pathlib import Path

import numpy as np
import pandas as pd

TIME_STEPS = 10


def read_data(path: Path):
    """
    Read raw data file.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.round("T")
    df = df.set_index("date")
    return df


def prepare_data_for_lstm(df, time_steps=1):
    """
    Prepare data for lstm model in required manner - (samples, time_steps, features)
    """
    y = df["Occupancy"]
    X = df.drop(columns=['Occupancy'])
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def create_training_test_files(src_path: Path, dst_path: Path, time_steps):
    """
    Read raw data and prepare it for ML model - then save it in desired place.
    """

    files = src_path.glob("*.txt")
    for file in files:
        if "training" in file.name:
            df = read_data(file)
            X, y = prepare_data_for_lstm(df, time_steps=time_steps)
            if dst_path.exists():
                np.save(dst_path / 'train_X.npy', X)
                np.save(dst_path / 'train_y.npy', y)
            else:
                dst_path.mkdir(exist_ok=True)
                np.save(dst_path / 'train_X.npy', X)
                np.save(dst_path / 'train_y.npy', y)
        elif "test" in file.name:
            df = read_data(file)
            X, y = prepare_data_for_lstm(df, time_steps=time_steps)
            if dst_path.exists():
                np.save(dst_path / 'test_X.npy', X)
                np.save(dst_path / 'test_y.npy', y)
            else:
                dst_path.mkdir(exist_ok=True)
                np.save(dst_path / 'test_X.npy', X)
                np.save(dst_path / 'test_y.npy', y)


def get_training_data(path: Path):
    """
    Read preprocessed, training lstm_data.
    """
    X = np.load(path / 'train_X.npy')
    y = np.load(path / 'train_y.npy')
    return X, y


def get_test_data(path: Path):
    """
    Read preprocessed, test lstm_data.
    """
    X = np.load(path / 'test_X.npy')
    y = np.load(path / 'test_y.npy')
    return X, y


if __name__ == "__main__":
    path = Path("../data/raw_data")
    dst_path = Path("../data/lstm_data")
    create_training_test_files(path, dst_path=dst_path, time_steps=TIME_STEPS)
    print(get_training_data(dst_path))
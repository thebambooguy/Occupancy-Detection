# Occupancy Detection 

### Introduction

A goal of this project is to learn more about Recurrent Neural Networks (RNN) and how it can be implemented with use of
Tensorflow and Keras API. It is a simple project that uses [Occupancy Detection](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+) 
dataset - dataset which consist of multivariate time series classification data to forecast occupancy of an office room. In this project ending 
score of the RNN is not so important - I would like to focus more on Tensorflow implementation and other ideas i.e.
creating ML pipeline with Airflow. Project is under development and uses python 3.7.

### Project structure

- code - code that preprocess data, create model and run experiments
    - model - models used during experiments
    - notebooks - ipynb notebooks which can help in data understanding
- data
    - lstm_data - preprocessed data for lstm model
    - raw_data - dataset downloaded from UCI
- results - training and test plots and scores which will help in better understading of the RNN architecture

### How to start
1. Run `pip install -r requiremnts.txt`.
2. Run `data.py` to generate data.
3. Run `train.py` to train model.
4. Run `test.py` to evaluate model.

#!/bin/bash

# Prepare data
python data.py

# Run training
python train.py --epochs 3 --batch_size 16

# Run test
python test.py
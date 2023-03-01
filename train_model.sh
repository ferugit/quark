#!/bin/bash
# Fernando López Gavilánez, 2023


# Set training parameters
optimizer='adam'
momentum=0.9
weight_decay=0.0
seed=2023
epochs=80
batch_size=12
patience=10
lr=0.001
cuda=True
balance=True
# arch
# augments

# Audio config
window_size=3.0
sampling_rate=16000

# Dataset
num_classes=5


python -u src/train_and_evaluate.py
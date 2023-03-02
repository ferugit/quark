#!/bin/bash
# Fernando López Gavilánez, 2023


# Audio dataset path
data_path='/home/wfl/Projects/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data'
partition_path='partitions/partition_20230302151711'

# Set training parameters
seed=2023
optimizer='adam'
momentum=0.9
weight_decay=0.0
epochs=80
batch_size=12
patience=10
lr=0.001
cuda=true
balance=True
# arch
# augments

# Audio config
window_size=3.0
sampling_rate=16000

# Dataset
num_classes=5


python -u src/train_and_evaluate.py --seed $seed --optimizer $optimizer \
-- momentum $momentum --weigth_decay $weight_decay --epochs $epochs --batch_size $batch_size \
--patience $patience --lr $lr --cuda $cuda --balance $balance --window_size $window_size \
--sampling_rate $sampling_rate --num_classes $num_classes --data_path $data_path \
--partition_path $partition_path
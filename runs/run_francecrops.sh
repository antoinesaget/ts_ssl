#!/bin/bash
set -e

dataset=francecrops_mini
model=simclr
scheduler=one_cycle
max_examples_seen=50000000
val_check_interval=1000
views_sampling_strategy=group
encoder=resnet
augmentations=resampling
post_training_eval=true

python ts_ssl/train.py model=$model \
    dataset=$dataset \
    optimizer/scheduler=$scheduler \
    training.max_examples_seen=$max_examples_seen training.val_check_interval=$val_check_interval \
    augmentations=$augmentations augmentations.views_sampling_strategy=$views_sampling_strategy \
    model/encoder=$encoder \
    post_training_eval=$post_training_eval

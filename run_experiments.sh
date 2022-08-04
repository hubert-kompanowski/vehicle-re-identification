#!/bin/bash

python train.py experiment=009_aug_one_cycle model.backbone=resnet50 name=aug_one_cycle_res50 datamodule.batch_size=16

sleep 10
python train.py experiment=009_aug_one_cycle



#!/bin/bash

python train.py experiment=012_mdl
sleep 10
python train.py experiment=013_mdl_res50
sleep 10
python train.py experiment=014_mdl_no_aug
sleep 10
python train.py experiment=015_mdl_res50_no_aug

#!/bin/bash

python train.py experiment=019_vehicle_id_res50

sleep 10
python train.py experiment=018_vehicle_id_ds

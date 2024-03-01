#!/bin/bash

missingratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# emumrate params
for param in "${missingratio[@]}"
do
python GradientInference.py --dataset awa2 --backbone resnet101_imagenet --freezebb False --lambda_xy 1 --lambda_xc 1 --lambda_cy 2 --intervene_type 'individual' --missingratio $param --trained_weight 
done
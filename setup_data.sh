#!/bin/bash

mkdir Keras_retinanet/snapshots
python ./Config/download_dataset_small.py

unzip ./dataset_small.zip 
rm ./dataset_small.zip

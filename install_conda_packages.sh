#!/bin/sh
conda activate trecx
#required for keyword spotting
conda install -c forge ffmpeg
conda install -c anaconda cudatoolkit=10.1
conda install -c anaconda cudnn=7.6
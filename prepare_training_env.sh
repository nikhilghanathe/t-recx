#!/bin/sh
conda create -n trecx python=3.7
conda activate trecx
#required for keyword spotting
conda install -c forge ffmpeg
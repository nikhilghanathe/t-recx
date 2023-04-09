
# T-RECX: Tiny-Resource Efficient Convolutional neural networks with early-eXit

Deploying Machine learning (ML) on the milliwatt-scale edge devices (tinyML) is gaining popularity due to recent breakthroughs in ML and Internet of Things (IoT). Most tinyML research focuses on model compression techniques that trade accuracy (and model capacity) for compact models to fit into the KB-sized tiny-edge devices. In this work, we show how such models can be enhanced by the addition of an early exit intermediate classifier. If the intermediate classifier exhibits sufficient confidence in its prediction, the network exits early thereby, resulting in considerable savings in time. Although early exit classifiers have been proposed in previous work, these previous proposals focus on large networks, making their techniques suboptimal/impractical for tinyML applications. Our technique is optimized specifically for tiny-CNN sized models. In addition, we present a method to alleviate the effect of network overthinking by leveraging the representations learned by the early exit. We study T-RecX on three CNNs from the [MLPerf tiny](https://mlcommons.org/en/inference-tiny-10/) benchmark suite for image classification, keyword spotting and visual wake word detection tasks. 

This repository contains code to reproduce results from "T-RECX: Tiny-Resource Efficient Convolutional neural networks with early-eXit" paper

```
@misc{ghanathe2022trecx,
      title={T-RECX: Tiny-Resource Efficient Convolutional Neural Networks with Early-Exit}, 
      author={Nikhil P Ghanathe and Steve Wilton},
      year={2022},
      eprint={2207.06613},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

**Requirements**
>* A Ubuntu Linux Machine with a GPU. Tested on GeForce RTX 2080 GPU 
>>* Works with Windows subsystem for Linux as well
>* Python 3.7, Tensorflow 2.3, Cudatoolkit 10.1, Cudnn 7.6

We evaluate three tinyML tasks from the MLPerf tiny benchmark suite.
>*  Image classification on CIFAR-10 with Resnet-8
>* Keyword Spotting on Speech Commands dataset with DSCNN
>* Visual Wake Words on Visual Wake word dataset with MobilenetV1

**Installation**

Install a package manger like [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Create a conda environment and install all software dependencies as follows.
```bash
conda create -n trecx python=3.7
conda activate trecx
bash install_conda_packages.sh
pip install -r requirements.txt
```
**Usage**

The repository contains three directories: `image_classification/`, `keyword_spotting/`, `visual_wake_words/`. These directories correspond  to evaluation of 1) Resnet-8 on CIFAR-10, 2) DSCNN on Speech Commands and 3) MobilenetV1 on Visual wake words datasets respectively. 
The scripts for downloading datasets, preprocessing, training and testing have all been automated, and are included in each directory.  See README.md in each directory for further instructions.

The repository also contains pretrained models. As an alternative to several hours of training, the results presented in this work can   be verified by running the test script included in each directory. Details on running the test scripts can be found in the corresponding README.md files.

**GPU Support**

For GPU support, you may need to customize your CUDA installation based on the version of tensorflow and CUDA drivers installed.  See [https://www.tensorflow.org/install/source#gpu](https://www.tensorflow.org/install/source#gpu) for more information.

  



# Keyword Spotting on Speech Commands Dataset

A DSCNN model is trained on the Speech Commands dataset and early-exit is applied to it.
The model distinguishes between 12 different classes on the Speech command dataset: 10 words + silence + unknown class

Model: DSCNN

Dataset: Speech Commands 

## Usage

Run the following commands to download dataset and reproduce results from the paper. This command will download the preprocess the dataset and store it to `data/`. Downloading + preprocessing will require 20-30min. Testing using pretrained models will require 10-15 min.
``` Bash
# Download dataset and test
python test_dscnn.py --data_dir=data/
```

To train the models,
```Bash
# Train DSCNN models (takes several hours)
./download_speech_cmd_train.sh 
```

## Description

The pretrained models can be found in `trained_models/`

The benefit curve presented in the paper are generated and saved to `results/` 

Running `./download_speech_cmd_train.sh`,
* Downloads the Google Speech Commands v2 data set to a directory (set by `--data_dir`, defaults to $HOME/data) after checking whether the data already exists there.  The data is structured as a TF dataset, not as individual wav files.
* Trains four versions of the DSCNN model ,
>* with Early-view (EV) assistance
>* without EV-assistance 
>* with [SDN](https://arxiv.org/abs/1810.07052) techniques 
>* with [Branchynet](https://arxiv.org/abs/1709.01686) techniques. 

Each model trains for around 4-5 hours with a GPU. The models are saved in `trained_models/`


Running `python test_dscnn.py --data_dir=data/` generates benefit curve plots using models from `trained_models/` and stores them in `results/`

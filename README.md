# Generating data using GAN on CIFAR-10 dataset

In this project we use the latest conditional GAN models to learn to generate images using the CIFAR-10 dataset. This dataset includes 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

As part of our project we take into consideration efficiency and training time. This means that we are careful to implement compact models that can be trained on a single GPU for a relatively short amount of time(less than two days).

## Set-Up
To set up our environment we use Anaconda. First create a new environment using the following command:
```
conda create -n environment_name python=3.9
```
We use Python 3.9 in our testing but other versions might be compatible. We use PyTorch as our main machine learning library which can be installed using the following commands:
```
# To install the cpu only version
conda install pytorch torchvision torchaudio cpuonly -c pytorch
# To install the nvidia/cuda version
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
We used PyTorch 1.10 and Cuda 10.2 in our testing but other version might be compatible. The other necessary packages can be installed using `pip` from the requirements file:
```
pip install -r requirements.txt
```
Alternatively you can create a conda environment with the name gan(can be changed manually in the variable name inside the `.yml` file), packed with all necessary packages using the `.yml` file:
```
conda env create --file environment.yml
```
This will also install the 'black' package for formatting code and 'mypy' for static typing in Python. You can run both using the commands:
```
# you can run both commands for a file like shown below or a directory
black file.py
mypy file.py
```

## Training
To start a training instance run the `src/train.py` program with the path to a valid(see `configs/default.yaml`) `.yaml` configuration file. If no such path is specified then the program will look for a `default.yaml` file under the `configs/` directory. If the configuration file is read correctly then a directory will be created to save the progress of the training using Tensorboard and a `.pt` file will be created to save the learned model. The first will be saved under `experiments/preset_name/current_time_key/` and the latter under `models/preset_name/current_time_key.pt`.

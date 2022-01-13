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
To set up our environment we use Anaconda. You can use one of two methods below.

- First create a new environment using the following command:
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

- Alternatively you can create a conda environment with the name gan(can be changed manually in the variable name inside the `.yml` file), packed with all necessary packages using the `.yml` file:
    ```
    conda env create --file environment.yml
    ```
    This is set by default to install the gpu version of pytorch.

Both methods will also install the 'black' package for formatting code and 'mypy' for static typing in Python. You can run both using the commands:
```
# you can run both commands for a file or directory like shown below
black file.py
mypy file.py
```

## Training
To start a training instance run the `src/train.py` program with the path to a valid(see `configs/default.yaml`) `.yaml` configuration file. If no such path is specified then the program will look for a `default.yaml` file under the `configs/` directory. If the configuration file is read correctly then a directory will be created to save the progress of the training using Tensorboard and `.pt` files will be created to save the generator and discriminator. The first will be saved under `experiments/config_name/current_time_key/` and the latter under `models/config_name/current_time_key/gen_or_disc.pt`. During training the loss values are tracked for both generator and discriminator on each epoch. The last condition to train the models properly is to create the statistics for the dataset in order to calculate the Frechet Inception distance(FID) which is explained in the next paragraph.

## Testing
Testing will be carried out at specific intervals set in the configuration. During training the training set is used to evaluate the loss for both generator and discriminator as well as the accuracy of the discriminator for real images and generated images. Some of the generated images are saved beside the real images for 'human observation' evaluation. During testing we also use Inception score(IS) and Frechet Inception distance(FID) to evaluate the generative model. These are calculated using the `pytorch_gan_metrics` library([git hub](https://github.com/w86763777/pytorch-gan-metrics)), by calling the function `get_inception_score_and_fid(gen_imgs, stats_path)`. Where `gen_imgs` refers to generated images and `stats_path` is the path of the precalculated statistics for the given dataset. To create the precalculated statistics first use `src/save_cifar10_imgs.py` to create a folder of all test images of CFIAR-10 then use the command:
```
python -m pytorch_gan_metrics.calc_fid_stats --path dataset/cifar10_images --output dataset/cifar10_fid_stats.npz
```
This will create a `.npz` file where the necessary data to calculate FID will be stored.

We have also implemented a `test_gen.py gnerator_path.pt config_path` program to generate images for a passed generator and configuration file as an argument.

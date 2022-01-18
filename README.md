# Generating data using GAN on CIFAR-10 dataset

In this project we use conditional GAN(DCGAN) models to learn to generate images using the CIFAR-10 dataset. This dataset includes 10 classes:

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

- Alternatively, you can create a conda environment with the name gan(can be changed manually in the variable name inside the `.yml` file) and install all necessary packages using the defined `environment.yml` file:
    ```
    conda env create --file environment.yml
    ```
    This is set by default to install the gpu version of pytorch.

Both methods will also install the 'black' package for formatting code and 'mypy'\* for static typing in Python. You can run both using the commands:
```
# you can run both commands for a file or directory like shown below
black file.py
mypy file.py
```

<sub>_*Not fully implemented through the code._

## Training
To start a training instance run the `src/train.py` program with the path to a valid(see `configs/default.yaml`) `.yaml` configuration file. If no such path is specified then the program will look for a `default.yaml` file under the `configs/` directory. If the configuration file is read correctly then a directory will be created to save the progress of the training using Tensorboard and `.pt` files will be created to save the generator and discriminator. The first will be saved under `experiments/config_name/current_time_key/` and the latter under `models/config_name/current_time_key/gen_or_disc.pt`. Furthermore, we also save the configuration file used to initialize the training under the `experiments/config_name/current_time_key/`, which comes in handy if small changes are made to the configuration before starting training. During training the loss values are tracked for both generator and discriminator on each epoch. The last condition to train the models properly is to create the statistics for the dataset in order to calculate the Frechet Inception distance(FID) which is explained in the next paragraph.

## Testing
Testing will be carried out at specific intervals set in the configuration. During testing the test set is used to evaluate the loss for both generator and discriminator as well as the accuracy of the discriminator for real images and generated images. Some of the generated images are saved beside the real images for 'human observation' evaluation. During testing we also use Inception score(IS) and Frechet Inception distance(FID) to evaluate the generative model. These are calculated using the `pytorch_gan_metrics` library([git hub](https://github.com/w86763777/pytorch-gan-metrics)), by calling the function `get_inception_score_and_fid(gen_imgs, stats_path)`. Where `gen_imgs` refers to generated images and `stats_path` is the path of the precalculated statistics for the given dataset. To create the precalculated statistics first use `src/save_cifar10_imgs.py` to create a folder of all test images of CFIAR-10 then use the command:
```
python -m pytorch_gan_metrics.calc_fid_stats --path dataset/cifar10_images --output dataset/cifar10_fid_stats.npz
```
This will create a `.npz` file where the necessary data to calculate FID will be stored. The FID stats are created only using test data as this is a more fair way to compare the generator to images that have not been used during training. Using the training data would almost definitely result in a slightly better FID.

We have also implemented a separate program for testing,
```
test_gen.py gnerator_path.pt config_path.yaml num_samples num_images
```
which generates, saves and evaluates images. You need to pass as arguments the path to a generator model and its corresponding configuration file. There is also the optional argument of passing the number of samples which is set by default to 1000. To get a fair evaluation for FID and IS scores you need to use at least 10000 samples. You can also pass the number of images you want to save which will be equally divided into the 10 possible classes, the default value is set to 100.

## Configuration
To run an experiment you can either use one of the existing configuration under the `configs/` folder or create your own. To help define a configuration you can use `default.yaml` configuration as template, where all possible options for each parameter are written. The default configuration trains a general conditional GAN. The `dcgan.yaml` configuration has been optimized for our task of interest, CIFAR10. The `wgan.yaml` and `wgan_gp.yaml` configuration both train a conditional Wasserstein GAN. The first applies clipping to the parameters of the discriminator whereas the second calculates a gradient penalty like proposed in this [paper](https://arxiv.org/pdf/1706.08500). Finally, the `deep_dcgan.yaml` uses feature matching loss and architectures developed based on this [work](https://arxiv.org/abs/1606.03498).

## Example and results
To run an experiment after setting up the environment, make also sure to have created the stats for the FID score.
```
# create folder with all test images under dataset/cifar10-images/
python save_cifar10_imgs.py

# create stats under dataset/cifar10_fid_stats.npz
python -m pytorch_gan_metrics.calc_fid_stats --path dataset/cifar10_images --output dataset/cifar10_fid_stats.npz

# run an experiment
python src/train.py configs/dcgan.py

# program output
Train model using configs/dcgan.yaml as configuration file.
Saving experiment under:         /path_to_repo/gan_cifar/experiments/dcgan/Tue_Jan_18_12_11_26_2022
Saving experiment models under:  /path_to_repo/gan_cifar/models/dcgan/Tue_Jan_18_12_11_26_2022
Files already downloaded and verified
Files already downloaded and verified
  0%|██                                                                            | 25/1000 [00:00<?, ?it/s]
  0%|████████████                                                                  | 55/391  [00:00<?, ?it/s]
# The progress over the epochs and the current training or testing epoch will be shown
# If the dataset is not downloaded it will be under the dataset/ directory

# To check the experiment results use tensorboard
tensorboard --logdir /path_to_repo/gan_cifar/experiment/dcgan/Tue_Jan_18_12_11_26_2022

# To test the generative model again use test_gen.py
python src/test_gen.py /path_to_repo/gan_cifar/models/dcgan/Tue_Jan_18_12_11_26_2022/gen.pt /path_to_repo/gan_cifar/experiments/dcgan/Tue_Jan_18_12_11_26_2022/dcgan.yaml 10000 100
```

### Performance results
Due to time and hardware constraints we have not yet tested on multiple seeds for each training/model configuration. Below are the best FID and IS scores\* achieved while testing. Some of the instances have been trained with older version of our code.

|  Models & Training \ Metrics      | FID(TEST 10k) |     IS       |
| :----                             | :----:        |     :----:   |
| DCGAN64                           |    39.94      |  6.56±0.15   |
| DCGAN128                          |    43.18      |  6.39±0.14   |
| DCGAN64_Batch_Dropout             |    52.15      |  6.23±0.17   |
| WGAN64_Instance                   |    46.36      |  6.17±0.14   |
| WGAN64_Layer                      |    49.21      |  5.90±0.13   |
| WGAN64_Batch                      |    77.05      |  4.95±0.15   |
| WGAN96_Instance                   |    63.32      |  5.28±0.11   |
| DEEPER_DCGAN16                    |    44.97      |  5.87±0.15   |
| DEEPER_DCGAN64                    |    65.22      |  5.39±0.11   |
| DEEPER_DCGAN46_Feat_Mat           |    81.72      |  4.43±0.07   |

<sub>_*For FID, lower is better. For IS, higher is better._

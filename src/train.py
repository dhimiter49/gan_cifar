import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import yaml
import nets

working_dir = Path(__file__).parent.parent.absolute()
unique_key = str(str(time.ctime())).replace(" ", "_")
experiments_dir = Path()  # set this paths after reading the config file
models_dir = Path()



def train(model, data_loader, optimizer, loss_function):
    pass


def test(model, data_loader):
    pass


def main():
    # read config file
    (
        batch_size,
        test_batch_size,
        epochs,
        lr,
        gamma,
        cuda,
        seed,
    ) = read_config(sys.argv)

    Path(experiments_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir.parent).mkdir(parents=True, exist_ok=True)
    open(models_dir, 'w+')

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")

    # load dataset
    # instantiate dataloader
    # instantiate network, optimizer, loss...
    # main loop, calls train and test
    # log results
    # plot stuff
    pass

  
def read_config(_input):
    if len(_input) == 1:
        print(
            "No configuration file specified.\n"
            "Looking for default configuration file under configs/ directory."
        )
        path_config = working_dir / Path("configs/default.yaml")
    else:
        print("Train model using %s as configuration file." % (_input[1]))
        path_config = working_dir / Path("{}".format(_input[1]))

    global experiments_dir, models_dir
    experiments_dir = working_dir / Path("experiments/" + path_config.stem) / Path(unique_key)
    models_dir = working_dir / Path("models/" + path_config.stem) / Path(unique_key + ".pt")

    if path_config.suffix != ".yaml":
        print("Make sure that the configuration file is a .yaml file.")
        sys.exit(1)

    try:
        with open(path_config) as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    except OSError as e:
        print("Configuration file not available, check under configs/ directory.")
        print(e)
        sys.exit(1)

    try:
        config = (
            batch_size,
            test_batch_size,
            epochs,
            lr,
            gamma,
            cuda,
            seed,
        ) = list(config["training"].values())
        assert type(batch_size) == int
        assert type(test_batch_size) == int
        assert type(epochs) == int
        assert type(lr) == float
        assert type(gamma) == float
        assert type(cuda) == bool
        assert type(seed) == int
    except (AssertionError, ValueError, KeyError) as e:
        print("The given .yaml file uses a wrong convention.")
        print(
            "The expected format for the .yaml file is:\n"
            "training:\n"
            "    batch_size: int\n"
            "    test_batch_size: int\n"
            "    epochs: int\n"
            "    lr: float\n"
            "    gamma: float\n"
            "    cuda: bool\n"
            "    seed: int"
        )
        print(e)
        sys.exit(1)

    return config


if __name__ == "__main__":
    main()

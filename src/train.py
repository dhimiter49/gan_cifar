import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import yaml
import nets

working_dir = Path(__file__).parent.parent.absolute()


def train(model, data_loader, optimizer, loss_function):
    pass


def test(model, data_loader):
    pass


def main():
    # read config file
    config = read_config(sys.argv)

    (
        batch_size,
        test_batch_size,
        epochs,
        lr,
        gamma,
        cuda,
        seed,
    ) = list(config["training"].values())

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

    try:
        with open(path_config) as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    except OSError as e:
        print("Configuration file not available, check under configs/ directory.")
        print(e)
        sys.exit(1)

    return config


if __name__ == "__main__":
    main()

import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import tqdm

import yaml
import nets
import losses

from torch.autograd import Variable

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

working_dir = Path(__file__).parent.parent.absolute()
unique_key = str(str(time.ctime())).replace(" ", "_")
experiments_dir = Path()  # set this paths after reading the config file
models_dir = Path()


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
        gen_loss_str,
        disc_loss_str,
    ) = read_config(sys.argv)

    # Path(experiments_dir).mkdir(parents=True, exist_ok=True)
    # Path(models_dir.parent).mkdir(parents=True, exist_ok=True)
    # open(models_dir, "w+")

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")

    # PyTorch transforms
    transform = transforms.Compose(
        [
            transforms.Resize((32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Read CIFAR10 data and apply transformation
    cifar10_dataset = torchvision.datasets.CIFAR10(
        root="./dataset", train=True, download=True, transform=transform
    )
    cifar10_dataset_test = torchvision.datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=transform
    )


    data_loader = torch.utils.data.DataLoader(
        cifar10_dataset, batch_size=1, shuffle=True, num_workers=1
    )
    data_loader_test = torch.utils.data.DataLoader(
        cifar10_dataset_test, batch_size=1, shuffle=False, num_workers=1
    )

    gen_loss = getattr(losses, gen_loss_str)()
    disc_loss = getattr(losses, disc_loss_str)()

    generator = DefaultGen()
    discriminator = DefaultDis()

    # Optimizer for the generator
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.999))

    # Optimizer for the discriminator
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.005)

    for epoch in range(epochs):
        # Create valid and fake labels as targets
        valid = Variable(torch.FloatTensor(len(data_loader), 1).fill_(1.0), requires_grad=False) #real
        fake = Variable(torch.FloatTensor(len(data_loader), 1).fill_(0.0), requires_grad=False) #fake

        real_imgs = Variable(data.type(torch.FloatTensor))

        # Create a random (sample from normal distribution) matrix as an input for the generator
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (len(data_loader), 128, 4, 4))))

        data_loader_z = torch.utils.data.DataLoader(
            z, batch_size=1, shuffle=True, num_workers=1)


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
    experiments_dir = (
        working_dir / Path("experiments/" + path_config.stem) / Path(unique_key)
    )
    models_dir = (
        working_dir / Path("models/" + path_config.stem) / Path(unique_key + ".pt")
    )

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
            gen_loss,
            disc_loss,
        ) = list(config["training"].values())
        assert type(batch_size) == int
        assert type(test_batch_size) == int
        assert type(epochs) == int
        assert type(lr) == float
        assert type(gamma) == float
        assert type(cuda) == bool
        assert type(seed) == int
        assert type(gen_loss) == str
        assert type(disc_loss) == str
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
            "    gen_loss: str"
            "    disc_loss: str"
        )
        print(e)
        sys.exit(1)

    return config


if __name__ == "__main__":
    main()

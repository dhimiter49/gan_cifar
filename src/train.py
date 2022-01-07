import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml

from nets import Discriminator, Generator
import losses

import os

os.environ["OMP_NUM_THREADS"] = "4"

# for windows
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

working_dir = Path(__file__).parent.parent.absolute()
unique_key = str(str(time.ctime())).replace(" ", "_").replace(":", "_")
experiments_dir = Path()  # set this paths after reading the config file
models_dir = Path()


def main():
    (
        IMG_SIZE,
        CHANNELS_IMG,
        NUM_CLASSES,
        DISC_FEATURES,
        GEN_FEATURES,
        LATENT_DIM,
        EMBEDDING_DIM,
        BATCH_SIZE,
        TEST_BATCH_SIZE,
        EPOCHS,
        LR,
        GAMMA,
        CUDA,
        SEED,
        DISC_ITERATIONS,
        GEN_LOSS_STR,
        DISC_LOSS_STR,
    ) = read_config(sys.argv)

    Path(experiments_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir.parent).mkdir(parents=True, exist_ok=True)
    open(models_dir, "w+")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if (CUDA and torch.cuda.is_available()) else "cpu")

    trans = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
            ),
        ]
    )

    # Read CIFAR10 data and apply transformation
    cifar10_dataset = torchvision.datasets.CIFAR10(
        root="./dataset", train=True, download=True, transform=trans
    )
    cifar10_dataset_test = torchvision.datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        cifar10_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    data_loader_test = torch.utils.data.DataLoader(
        cifar10_dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4
    )

    gen_loss = getattr(losses, GEN_LOSS_STR)()
    disc_loss = getattr(losses, DISC_LOSS_STR)()

    generator = Generator(
        LATENT_DIM, CHANNELS_IMG, GEN_FEATURES, NUM_CLASSES, IMG_SIZE, EMBEDDING_DIM
    ).to(device)
    discriminator = Discriminator(
        CHANNELS_IMG, DISC_FEATURES, NUM_CLASSES, IMG_SIZE
    ).to(device)

    writer_real = SummaryWriter(experiments_dir / Path("real"))
    writer_fake = SummaryWriter(experiments_dir / Path("fake"))
    step = 0

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.9, 0.999))
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=LR, betas=(0.5, 0.999), weight_decay=0.005
    )

    for epoch in tqdm(range(EPOCHS)):
        for batch_idx, (data, labels) in enumerate(tqdm(data_loader)):
            data, labels = data.to(device), labels.to(device)
            mini_batch_size = data.shape[0]
            real_targets = torch.ones(mini_batch_size).to(device)
            fake_targets = torch.zeros(mini_batch_size).to(device)

            for _ in range(DISC_ITERATIONS):
                noise = torch.randn(mini_batch_size, LATENT_DIM, 1, 1).to(device)
                fake = generator(noise, labels)
                predicition_real = discriminator(data, labels).view(-1)
                predicition_fake = discriminator(fake, labels).view(-1)
                loss_real = disc_loss(predicition_real, real_targets)
                loss_fake = disc_loss(predicition_fake, fake_targets)
                loss_disc = loss_real + loss_fake
                discriminator.zero_grad()
                loss_disc.backward(retain_graph=True)
                disc_optimizer.step()

            predicition_fake = discriminator(fake, labels).view(-1)
            loss_gen = gen_loss(predicition_fake, real_targets)
            generator.zero_grad()
            loss_gen.backward()
            gen_optimizer.step()


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
        config_dataset = (
            img_size,
            channels_img,
            num_classes,
        ) = list(config["dataset"].values())

        config_model = (
            disc_features,
            gen_features,
            latent_dim,
            embedding_dim,
        ) = list(config["nets"].values())

        config_training = (
            batch_size,
            test_batch_size,
            epochs,
            lr,
            gamma,
            cuda,
            seed,
            disc_iterations,
            gen_loss,
            disc_loss,
        ) = list(config["training"].values())

        assert type(img_size) == int
        assert type(channels_img) == int
        assert type(num_classes) == int
        assert type(disc_features) == int
        assert type(gen_features) == int
        assert type(latent_dim) == int
        assert type(embedding_dim) == int
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
            "dataset:\n"
            "    img_size: int\n"
            "    channels_img: int\n"
            "    num_classes: int\n"
            "nets:\n"
            "    disc_features: int\n"
            "    gen_features: int\n"
            "    latent_dims: int\n"
            "    embedding_dim: int\n"
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

    return config_dataset + config_model + config_training


if __name__ == "__main__":
    main()

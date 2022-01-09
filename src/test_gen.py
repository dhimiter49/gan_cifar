import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import yaml

from nets import Generator

working_dir = Path(__file__).parent.parent.absolute()
experiments_dir = Path()  # set this paths after reading the config file
str_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def main():
    (
        PATH_MODEL,
        IMG_SIZE,
        CHANNELS_IMG,
        NUM_CLASSES,
        GEN_FEATURES,
        LATENT_DIM,
        EMBEDDING_DIM,
        TEST_BATCH_SIZE,
        CUDA,
    ) = read_config(sys.argv)

    Path(experiments_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (CUDA and torch.cuda.is_available()) else "cpu")
    generator = Generator(
        LATENT_DIM, CHANNELS_IMG, GEN_FEATURES, NUM_CLASSES, IMG_SIZE, EMBEDDING_DIM
    )
    generator.load_state_dict(torch.load(PATH_MODEL, map_location=torch.device(device)))
    generator.eval()

    n_img_per_class = int(TEST_BATCH_SIZE / NUM_CLASSES)
    noise = torch.randn(n_img_per_class * NUM_CLASSES, LATENT_DIM, 1, 1).to(device)
    labels = [
        torch.zeros(n_img_per_class, dtype=torch.int) + i for i in range(NUM_CLASSES)
    ]
    labels = torch.cat(labels)
    fake = generator(noise, labels)
    for i, img in enumerate(fake):
        img = img / 2 + 0.5
        save_image(
            img,
            experiments_dir
            / Path(str(i) + "_" + str_labels[labels[i].item()] + "_fake.png"),
        )


def read_config(_input):
    path_model = working_dir / Path(_input[1])
    path_config = working_dir / Path(_input[2])

    try:
        with open(path_config) as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    except OSError as e:
        print("Configuration file not available, check under configs/ directory.")
        print(e)
        sys.exit(1)

    global experiments_dir
    experiments_dir = (
        working_dir
        / Path("experiments/" + path_config.stem)
        / Path(path_model.stem.removesuffix("_gen"))
    )

    try:
        config_dataset = (
            img_size,
            channels_img,
            num_classes,
        ) = list(config["dataset"].values())

        config_model = (gen_features, latent_dim, embedding_dim,) = list(
            config["nets"].values()
        )[1:]

        test_batch_size = config["training"]["test_batch_size"]
        cuda = config["training"]["cuda"]

        assert type(img_size) == int
        assert type(channels_img) == int
        assert type(num_classes) == int
        assert type(gen_features) == int
        assert type(latent_dim) == int
        assert type(embedding_dim) == int
        assert type(cuda) == bool
        assert type(test_batch_size) == int
    except (AssertionError, ValueError, KeyError) as e:
        print("The given .yaml file uses a wrong convention.")
        print(e)
        sys.exit(1)

    return [path_model] + config_dataset + config_model + [test_batch_size, cuda]


if __name__ == "__main__":
    main()

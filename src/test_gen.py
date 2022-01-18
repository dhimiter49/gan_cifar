import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import yaml
from pytorch_gan_metrics import get_inception_score_and_fid

import nets


working_dir = Path(__file__).parent.parent.absolute()
experiments_dir = Path()  # set the path after reading the config file
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
        N_SAMPLES,
        N_IMGS,
        IMG_SIZE,
        CHANNELS_IMG,
        NUM_CLASSES,
        GENERATOR_MODEL,
        GEN_FEATURES,
        LATENT_DIM,
        EMBEDDING_DIM,
        GEN_NORMALIZERS,
        TEST_BATCH_SIZE,
        CUDA,
    ) = read_config(sys.argv)

    N_SAMPLES = max(N_SAMPLES, N_IMGS)
    Path(experiments_dir).mkdir(parents=True, exist_ok=True)
    print("Saving generated images under: \t", experiments_dir)

    device = torch.device("cuda" if (CUDA and torch.cuda.is_available()) else "cpu")

    gen = getattr(nets, GENERATOR_MODEL)(
        LATENT_DIM,
        CHANNELS_IMG,
        GEN_FEATURES,
        NUM_CLASSES,
        IMG_SIZE,
        EMBEDDING_DIM,
        GEN_NORMALIZERS,
    ).to(device)
    gen.load_state_dict(torch.load(PATH_MODEL, map_location=torch.device(device)))
    gen.eval()

    LATENT_MATRIX = 1
    if "4x4" in GENERATOR_MODEL:
        LATENT_MATRIX = 4

    N_SAMPLES = (N_SAMPLES // NUM_CLASSES) * NUM_CLASSES
    n_img_per_class = N_SAMPLES // NUM_CLASSES
    noise = torch.randn(
        n_img_per_class * NUM_CLASSES, LATENT_DIM, LATENT_MATRIX, LATENT_MATRIX
    ).to(device)
    labels = [
        torch.zeros(n_img_per_class, dtype=torch.int) + i for i in range(NUM_CLASSES)
    ]
    labels = torch.cat(labels).to(device)
    labels = labels[torch.randperm(len(labels))]
    input_loader = torch.utils.data.DataLoader(noise, batch_size=TEST_BATCH_SIZE)
    fake = torch.zeros(N_SAMPLES, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        for i, z in enumerate(tqdm(input_loader)):
            start_idx = i * TEST_BATCH_SIZE
            end_idx = min((i + 1) * TEST_BATCH_SIZE, N_SAMPLES)
            fake[start_idx:end_idx] = gen(z, labels[start_idx:end_idx])
    for i, img in enumerate(fake):
        if i >= N_IMGS:
            break
        img = img / 2 + 0.5
        save_image(
            img,
            experiments_dir
            / Path(str(i) + "_" + str_labels[labels[i].item()] + "_fake.png"),
        )

    (incep_score, incep_score_std,), frechet_distance = get_inception_score_and_fid(
        fake / 2 + 0.5, working_dir / Path("dataset/cifar10_fid_stats.npz")
    )
    print("Inception score: \t\t", incep_score)
    print("Inception score std: \t\t", incep_score_std)
    print("Frechet Inception distance: \t", frechet_distance)


def read_config(_input):
    path_model = working_dir / Path(_input[1])
    path_config = working_dir / Path(_input[2])
    n_samples = int(_input[3]) if len(_input) >= 4 else 1000
    n_imgs = int(_input[4]) if len(_input) >= 5 else 100

    try:
        with open(path_config) as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    except OSError as e:
        print("Configuration file not available, check under configs/ directory.")
        print(e)
        sys.exit(1)

    global experiments_dir
    experiments_dir = working_dir / Path(
        "experiments/" + path_config.stem + "/" + path_config.parent.stem
    )

    try:
        config_dataset = (
            img_size,
            channels_img,
            num_classes,
        ) = list(config["dataset"].values())

        gen_model = config["nets"]["generator_model"]
        gen_features = config["nets"]["gen_features"]
        latent_dim = config["nets"]["latent_dim"]
        embedding_dim = config["nets"]["embedding_dim"]
        gen_norm = config["nets"]["gen_normalizers"]
        config_model = [gen_model, gen_features, latent_dim, embedding_dim, gen_norm]

        test_batch_size = config["training"]["test_batch_size"]
        cuda = config["training"]["cuda"]

        assert type(img_size) == int
        assert type(channels_img) == int
        assert type(num_classes) == int
        assert type(gen_model) == str
        assert type(gen_features) == int
        assert type(latent_dim) == int
        assert type(embedding_dim) == int
        assert type(gen_norm) == list
        assert type(cuda) == bool
        assert type(test_batch_size) == int
    except (AssertionError, ValueError, KeyError) as e:
        print("The given .yaml file uses a wrong convention.")
        print(e)
        sys.exit(1)

    return (
        [path_model, n_samples, n_imgs]
        + config_dataset
        + config_model
        + [test_batch_size, cuda]
    )


if __name__ == "__main__":
    main()

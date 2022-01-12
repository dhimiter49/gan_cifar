import sys
import time
from pathlib import Path
import yaml
import torch
import torch.nn as nn


working_dir = Path(__file__).parent.parent.absolute()
unique_key = str(str(time.ctime())).replace(" ", "_").replace(":", "_")
experiments_dir = Path()  # set this paths after reading the config file
gen_dir = Path()
disc_dir = Path()

def gradient_penalty(discriminator, labels, data, fake, device="cpu"):
    BATCH_SIZE, C, H, W = data.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = data * epsilon + fake * (1 - epsilon)

    # Calculate critic scores
    mixed_scores = discriminator(interpolated_images, labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)  # l2 norm
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def initialize_weights(model):
    # According to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


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

    global experiments_dir, gen_dir, disc_dir
    experiments_dir = (
        working_dir / Path("experiments/" + path_config.stem) / Path(unique_key)
    )
    gen_dir = (
        working_dir
        / Path("models/" + path_config.stem)
        / Path(unique_key)
        / Path("gen.pt")
    )
    disc_dir = (
        working_dir
        / Path("models/" + path_config.stem)
        / Path(unique_key)
        / Path("disc.pt")
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
            generator_model,
            discriminator_model,
            disc_features,
            gen_features,
            latent_dim,
            embedding_dim,
        ) = list(config["nets"].values())

        config_training = (
            batch_size,
            test_batch_size,
            test_every,
            save_every,
            epochs,
            gen_lr,
            disc_lr,
            cuda,
            seed,
            disc_iterations,
            weight_clip,
            lambda_gp,
            gen_loss,
            disc_loss,
        ) = list(config["training"].values())

        assert type(img_size) == int
        assert type(channels_img) == int
        assert type(num_classes) == int
        assert type(generator_model) == str
        assert type(discriminator_model) == str
        assert type(disc_features) == int
        assert type(gen_features) == int
        assert type(latent_dim) == int
        assert type(embedding_dim) == int
        assert type(batch_size) == int
        assert type(test_batch_size) == int
        assert type(test_every) == int
        assert type(save_every) == int
        assert type(epochs) == int
        assert type(gen_lr) == float
        assert type(disc_lr) == float
        assert type(cuda) == bool
        assert type(seed) == int
        assert type(disc_iterations) == int
        assert type(weight_clip) == float
        assert type(lambda_gp) == int
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
            "    generator_model: str\n"
            "    discriminator_model: str\n"
            "    disc_features: int\n"
            "    gen_features: int\n"
            "    latent_dims: int\n"
            "    embedding_dim: int\n"
            "training:\n"
            "    batch_size: int\n"
            "    test_batch_size: int\n"
            "    test_every: int\n"
            "    save_every: int\n"
            "    epochs: int\n"
            "    gen_lr: float\n"
            "    disc_lr: float\n"
            "    cuda: bool\n"
            "    seed: int"
            "    disc_iterations: int"
            "    weight_clip: float"
            "    lambda_gp: int"
            "    gen_loss: str"
            "    disc_loss: str"
        )
        print(e)
        sys.exit(1)

    dirs = [experiments_dir, gen_dir, disc_dir]
    return config_dataset + config_model + config_training + dirs

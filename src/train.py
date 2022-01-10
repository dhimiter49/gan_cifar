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
from pytorch_gan_metrics import get_inception_score_and_fid
import yaml

import nets

from utils import gradient_penalty, initialize_weights

import losses

import os

os.environ["OMP_NUM_THREADS"] = "4"

# for windows
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

working_dir = Path(__file__).parent.parent.absolute()
unique_key = str(str(time.ctime())).replace(" ", "_").replace(":", "_")
experiments_dir = Path()  # set this paths after reading the config file
gen_dir = Path()
disc_dir = Path()


def main():
    (
        IMG_SIZE,
        CHANNELS_IMG,
        NUM_CLASSES,
        GENERATOR_MODEL,
        DISCRIMINATOR_MODEL,
        DISC_FEATURES,
        GEN_FEATURES,
        LATENT_DIM,
        EMBEDDING_DIM,
        BATCH_SIZE,
        TEST_BATCH_SIZE,
        TEST_EVERY,
        SAVE_EVERY,
        EPOCHS,
        GEN_LR,
        DISC_LR,
        GAMMA,
        CUDA,
        SEED,
        DISC_ITERATIONS,
        WEIGHT_CLIP,
        LAMBDA_GP,
        GEN_LOSS_STR,
        DISC_LOSS_STR,
    ) = read_config(sys.argv)

    Path(experiments_dir).mkdir(parents=True, exist_ok=True)
    Path(gen_dir.parent).mkdir(parents=True, exist_ok=True)
    open(gen_dir, "w+")
    open(disc_dir, "w+")
    print("Saving experiment under: \t", experiments_dir)
    print("Saving experiment models under: ", gen_dir.parent)

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
    N_TRAIN_DATA = len(cifar10_dataset)
    N_TEST_DATA = len(cifar10_dataset_test)

    data_loader = torch.utils.data.DataLoader(
        cifar10_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    data_loader_test = torch.utils.data.DataLoader(
        cifar10_dataset_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4
    )

    gen_loss = getattr(losses, GEN_LOSS_STR)()
    disc_loss = getattr(losses, DISC_LOSS_STR)()

    generator = getattr(nets, GENERATOR_MODEL)(
        LATENT_DIM, CHANNELS_IMG, GEN_FEATURES, NUM_CLASSES, IMG_SIZE, EMBEDDING_DIM
    ).to(device)

    discriminator = getattr(nets, DISCRIMINATOR_MODEL)(
        CHANNELS_IMG, DISC_FEATURES, NUM_CLASSES, IMG_SIZE
    ).to(device)

    initialize_weights(generator)
    initialize_weights(discriminator)

    writer = SummaryWriter(experiments_dir)

    gen_optimizer = torch.optim.Adam(
        generator.parameters(), lr=GEN_LR, betas=(0.9, 0.999)
    )
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=DISC_LR, betas=(0.5, 0.999), weight_decay=0.005
    )

    real_factor = 1
    fake_factor = 0
    if GEN_LOSS_STR == "WassersteinLoss" and DISC_LOSS_STR == "WassersteinLoss":
        real_factor = -1
        fake_factor = 1

    for epoch in tqdm(range(EPOCHS)):
        generator.train()
        discriminator.train()
        epoch_loss_disc = 0
        epoch_loss_gen = 0
        for (data, labels) in tqdm(data_loader, leave=False):
            data, labels = data.to(device), labels.to(device)
            mini_batch_size = data.shape[0]

            real_targets = real_factor * torch.ones(mini_batch_size).to(device)
            fake_targets = fake_factor * torch.ones(mini_batch_size).to(device)

            batch_loss_disc = []
            for _ in range(DISC_ITERATIONS):
                noise = torch.randn(mini_batch_size, LATENT_DIM, 1, 1).to(device)
                fake = generator(noise, labels)
                prediction_real = discriminator(data, labels).view(-1)
                prediction_fake = discriminator(fake, labels).view(-1)
                loss_real = disc_loss(prediction_real, real_targets)
                loss_fake = disc_loss(prediction_fake, fake_targets)

                gp = 0.0
                if LAMBDA_GP != 0:
                    gp = gradient_penalty(
                        discriminator, labels, data, fake, device=device
                    )

                loss_disc = (loss_real + loss_fake) + LAMBDA_GP * gp
                batch_loss_disc.append(loss_disc.item())
                discriminator.zero_grad()
                loss_disc.backward(retain_graph=True)
                disc_optimizer.step()

                if WEIGHT_CLIP != 0.0:
                    for p in discriminator.parameters():
                        p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            epoch_loss_disc += np.mean(batch_loss_disc)

            prediction_fake = discriminator(fake, labels).view(-1)
            loss_gen = gen_loss(prediction_fake, real_targets)
            epoch_loss_gen += loss_gen.item()
            generator.zero_grad()
            loss_gen.backward()
            gen_optimizer.step()

        step = epoch + 1
        epoch_loss_disc = epoch_loss_disc / N_TRAIN_DATA
        epoch_loss_gen = epoch_loss_gen / N_TRAIN_DATA
        writer.add_scalar("train_loss/discriminator", epoch_loss_disc, step)
        writer.add_scalar("train_loss/generator", epoch_loss_gen, step)

        if step % TEST_EVERY == 0:
            generator.eval()
            discriminator.eval()
            epoch_loss_disc = 0
            epoch_loss_gen = 0
            accuracy_real = 0.0
            accuracy_fake = 0.0
            incep_score = 0.0
            incep_score_std = 0.0
            frechet_distance = 0.0
            n_imgs_epoch = 50
            n_imgs = int(N_TEST_DATA / TEST_BATCH_SIZE) * n_imgs_epoch
            imgs_fake = torch.zeros(n_imgs, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
            imgs_real = torch.zeros(n_imgs, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
            for idx, (data, labels) in enumerate(tqdm(data_loader_test, leave=False)):
                data, labels = data.to(device), labels.to(device)
                mini_batch_size = data.shape[0]
                real_targets = torch.ones(mini_batch_size).to(device)
                fake_targets = torch.zeros(mini_batch_size).to(device)

                noise = torch.randn(mini_batch_size, LATENT_DIM, 1, 1).to(device)
                fake = generator(noise, labels)
                prediction_real = discriminator(data, labels).view(-1)
                prediction_fake = discriminator(fake, labels).view(-1)
                loss_real = disc_loss(prediction_real, real_targets)
                loss_fake = disc_loss(prediction_fake, fake_targets)
                loss_disc = loss_real + loss_fake
                epoch_loss_disc += loss_disc.item()
                loss_gen = gen_loss(prediction_fake, real_targets)
                epoch_loss_gen += loss_gen.item()

                # inception score and frechet inception distance
                (i_s, i_s_std), fid = get_inception_score_and_fid(
                    fake / 2 + 0.5, working_dir / Path("dataset/cifar10_fid_stats.npz")
                )
                incep_score += i_s
                incep_score_std += i_s_std
                frechet_distance += fid

                # save random real/fake images
                random_indexes = np.random.choice(
                    TEST_BATCH_SIZE, size=n_imgs_epoch, replace=False
                )
                start_idx = idx * n_imgs_epoch
                end_idx = start_idx + n_imgs_epoch
                imgs_real[start_idx:end_idx] = data[random_indexes]
                imgs_fake[start_idx:end_idx] = fake[random_indexes]

                # accuracy
                prediction_fake = prediction_fake >= 0.5
                prediction_real = prediction_real >= 0.5
                batch_correct_fake_pred = prediction_fake.eq(fake_targets).sum().item()
                batch_correct_real_pred = prediction_real.eq(real_targets).sum().item()
                accuracy_fake += batch_correct_fake_pred
                accuracy_real += batch_correct_real_pred

            # tracking
            epoch_loss_disc = epoch_loss_disc / N_TEST_DATA
            epoch_loss_gen = epoch_loss_gen / N_TEST_DATA
            accuracy_fake = accuracy_fake / N_TEST_DATA
            accuracy_real = accuracy_real / N_TEST_DATA
            incep_score = incep_score / (N_TEST_DATA / TEST_BATCH_SIZE)
            incep_score_std = incep_score_std / (N_TEST_DATA / TEST_BATCH_SIZE)
            frechet_distance = frechet_distance / (N_TEST_DATA / TEST_BATCH_SIZE)
            writer.add_scalar("test_loss/discriminator", epoch_loss_disc, step)
            writer.add_scalar("test_loss/generator", epoch_loss_gen, step)
            writer.add_scalar("test_accuracy/real", accuracy_real, step)
            writer.add_scalar("test_accuracy/fake", accuracy_fake, step)
            writer.add_scalar("evaluation/inception_score", incep_score, step)
            writer.add_scalar("evaluation/inception_std", incep_score_std, step)
            writer.add_scalar("evaluation/test_frechet_distance", frechet_distance, step)
            grid_real = torchvision.utils.make_grid(imgs_real, nrow=16, normalize=True)
            grid_fake = torchvision.utils.make_grid(imgs_fake, nrow=16, normalize=True)
            writer.add_image("real", grid_real, step)
            writer.add_image("fake", grid_fake, step)

        if step % SAVE_EVERY == 0:
            torch.save(generator.state_dict(), gen_dir)
            torch.save(discriminator.state_dict(), disc_dir)


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
            gamma,
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
        assert type(gamma) == float
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

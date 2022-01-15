import os
import sys
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

import nets
import losses
from utils import gradient_penalty, initialize_weights, read_config

# for windows
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["OMP_NUM_THREADS"] = "4"
working_dir = Path(__file__).parent.parent.absolute()


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
        DISC_NORMALIZERS,
        GEN_NORMALIZERS,
        BATCH_SIZE,
        TEST_BATCH_SIZE,
        TEST_EVERY,
        SAVE_EVERY,
        EPOCHS,
        GEN_LR,
        DISC_LR,
        CUDA,
        SEED,
        DISC_ITERATIONS,
        WEIGHT_CLIP,
        LAMBDA_GP,
        GEN_LOSS_STR,
        DISC_LOSS_STR,
        EXPERIMENT_DIR,
        GEN_DIR,
        DISC_DIR,
    ) = read_config(sys.argv)

    Path(EXPERIMENT_DIR).mkdir(parents=True, exist_ok=True)
    Path(GEN_DIR.parent).mkdir(parents=True, exist_ok=True)
    open(GEN_DIR, "w+")
    open(GEN_DIR.parent / Path("gen_best.pt"), "w+")
    open(DISC_DIR, "w+")
    open(DISC_DIR.parent / Path("disc_best.pt"), "w+")
    print("Saving experiment under: \t", EXPERIMENT_DIR)
    print("Saving experiment models under: ", GEN_DIR.parent)

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

    gen = getattr(nets, GENERATOR_MODEL)(
        LATENT_DIM,
        CHANNELS_IMG,
        GEN_FEATURES,
        NUM_CLASSES,
        IMG_SIZE,
        EMBEDDING_DIM,
        GEN_NORMALIZERS,
    ).to(device)

    disc = getattr(nets, DISCRIMINATOR_MODEL)(
        CHANNELS_IMG,
        DISC_FEATURES,
        NUM_CLASSES,
        IMG_SIZE,
        DISC_NORMALIZERS,
    ).to(device)

    LATENT_MATRIX = 1
    if "4x4" in GENERATOR_MODEL:
        LATENT_MATRIX = 4

    gen_loss = getattr(losses, GEN_LOSS_STR)()
    disc_loss = getattr(losses, DISC_LOSS_STR)()

    real_factor = 1
    fake_factor = 0
    if GEN_LOSS_STR == "WassersteinLoss" and DISC_LOSS_STR == "WassersteinLoss":
        real_factor = -1
        fake_factor = 1
        initialize_weights(gen)
        initialize_weights(disc)

    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=GEN_LR, betas=(0.5, 0.9))
    disc_optimizer = torch.optim.Adam(
        disc.parameters(), lr=DISC_LR, betas=(0.5, 0.9), weight_decay=0.005
    )

    writer = SummaryWriter(EXPERIMENT_DIR)
    best_FID_IS_score = 0.0

    for epoch in tqdm(range(EPOCHS)):
        gen.train()
        disc.train()
        epoch_loss_disc = 0.0
        epoch_loss_gen = 0.0
        for (data, labels) in tqdm(data_loader, leave=False):
            data, labels = data.to(device), labels.to(device)
            mini_batch_size = data.shape[0]
            real_targets = real_factor * torch.ones(mini_batch_size).to(device)
            fake_targets = fake_factor * torch.ones(mini_batch_size).to(device)
            if GEN_LOSS_STR == "BCELoss":  # label smoothing
                real_targets *= (
                    torch.ones(mini_batch_size).uniform_(0.7, 0.9).to(device)
                )

            batch_loss_disc = []
            for _ in range(DISC_ITERATIONS):
                disc.zero_grad()
                noise = torch.randn(
                    mini_batch_size, LATENT_DIM, LATENT_MATRIX, LATENT_MATRIX
                ).to(device)
                fake = gen(noise, labels)
                prediction_real = disc(data, labels).view(-1)
                prediction_fake = disc(fake, labels).view(-1)
                loss_real = disc_loss(prediction_real, real_targets)
                loss_fake = disc_loss(prediction_fake, fake_targets)

                gp = 0.0
                if LAMBDA_GP != 0:
                    gp = gradient_penalty(disc, labels, data, fake, device)

                loss_disc = (loss_real + loss_fake) + LAMBDA_GP * gp
                batch_loss_disc.append(loss_disc.item())
                loss_disc.backward(retain_graph=True)
                disc_optimizer.step()

                if WEIGHT_CLIP != 0.0:
                    for p in disc.parameters():
                        p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

            epoch_loss_disc += np.mean(batch_loss_disc)

            gen.zero_grad()
            noise = torch.randn(
                mini_batch_size, LATENT_DIM, LATENT_MATRIX, LATENT_MATRIX
            ).to(device)
            fake = gen(noise, labels)
            prediction_fake = disc(fake, labels).view(-1)
            loss_gen = gen_loss(prediction_fake, real_targets)
            epoch_loss_gen += loss_gen.item()
            loss_gen.backward()
            gen_optimizer.step()

        step = epoch + 1
        epoch_loss_disc = epoch_loss_disc / N_TRAIN_DATA
        epoch_loss_gen = epoch_loss_gen / N_TRAIN_DATA
        writer.add_scalar("train_loss/discriminator", epoch_loss_disc, step)
        writer.add_scalar("train_loss/generator", epoch_loss_gen, step)

        if step % TEST_EVERY == 0:
            gen.eval()
            disc.eval()
            epoch_loss_disc = 0.0
            epoch_loss_gen = 0.0
            accuracy_real = 0.0
            accuracy_fake = 0.0
            incep_score = 0.0
            incep_score_std = 0.0
            frechet_distance = 0.0
            n_imgs_epoch = TEST_BATCH_SIZE // 50  # save around 2%(1/50) of TEST DATASET
            n_imgs = (
                N_TEST_DATA // TEST_BATCH_SIZE
            ) * n_imgs_epoch + N_TEST_DATA % TEST_BATCH_SIZE
            imgs_fake = torch.zeros(n_imgs, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
            imgs_real = torch.zeros(n_imgs, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
            all_fakes = torch.zeros(N_TEST_DATA, CHANNELS_IMG, IMG_SIZE, IMG_SIZE)
            for idx, (data, labels) in enumerate(tqdm(data_loader_test, leave=False)):
                data, labels = data.to(device), labels.to(device)
                mini_batch_size = data.shape[0]

                real_targets = real_factor * torch.ones(mini_batch_size).to(device)
                fake_targets = fake_factor * torch.ones(mini_batch_size).to(device)

                noise = torch.randn(
                    mini_batch_size, LATENT_DIM, LATENT_MATRIX, LATENT_MATRIX
                ).to(device)
                fake = gen(noise, labels)
                prediction_real = disc(data, labels).view(-1)
                prediction_fake = disc(fake, labels).view(-1)
                loss_real = disc_loss(prediction_real, real_targets)
                loss_fake = disc_loss(prediction_fake, fake_targets)
                loss_disc = loss_real + loss_fake
                epoch_loss_disc += loss_disc.item()
                loss_gen = gen_loss(prediction_fake, real_targets)
                epoch_loss_gen += loss_gen.item()
                all_fakes[idx * TEST_BATCH_SIZE : (idx + 1) * TEST_BATCH_SIZE] = fake

                # save random real/fake images
                start_idx = idx * n_imgs_epoch
                n_imgs_epoch = min(n_imgs_epoch, mini_batch_size)
                random_indexes = np.random.choice(
                    mini_batch_size, size=n_imgs_epoch, replace=False
                )
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

            # inception score and frechet inception distance
            (
                incep_score,
                incep_score_std,
            ), frechet_distance = get_inception_score_and_fid(
                all_fakes / 2 + 0.5, working_dir / Path("dataset/cifar10_fid_stats.npz")
            )

            # tracking
            epoch_loss_disc = epoch_loss_disc / N_TEST_DATA
            epoch_loss_gen = epoch_loss_gen / N_TEST_DATA
            accuracy_fake = accuracy_fake / N_TEST_DATA
            accuracy_real = accuracy_real / N_TEST_DATA
            writer.add_scalar("test_loss/discriminator", epoch_loss_disc, step)
            writer.add_scalar("test_loss/generator", epoch_loss_gen, step)
            writer.add_scalar("test_accuracy/real", accuracy_real, step)
            writer.add_scalar("test_accuracy/fake", accuracy_fake, step)
            writer.add_scalar("evaluation/inception_score", incep_score, step)
            writer.add_scalar("evaluation/inception_std", incep_score_std, step)
            writer.add_scalar("evaluation/frechet_distance", frechet_distance, step)
            grid_real = torchvision.utils.make_grid(imgs_real, nrow=16, normalize=True)
            grid_fake = torchvision.utils.make_grid(imgs_fake, nrow=16, normalize=True)
            writer.add_image("real", grid_real, step)
            writer.add_image("fake", grid_fake, step)
            if frechet_distance / 10 + incep_score > best_FID_IS_score:
                torch.save(gen.state_dict(), GEN_DIR.parent / Path("gen_best.pt"))
                torch.save(disc.state_dict(), DISC_DIR.parent / Path("disc_best.pt"))

        if step % SAVE_EVERY == 0:
            torch.save(gen.state_dict(), GEN_DIR)
            torch.save(disc.state_dict(), DISC_DIR)


if __name__ == "__main__":
    main()

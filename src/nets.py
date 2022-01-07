import torch
import torch.nn as nn
import torch.nn.functional as F


## DCGAN
class Generator(nn.Module):
    def __init__(
        self,
        latent_dim,
        channels_img,
        gen_features,
        num_classes,
        img_size,
        embedding_dim,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                latent_dim + embedding_dim,
                gen_features * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(gen_features * 8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                gen_features * 8,
                gen_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(gen_features * 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                gen_features * 4,
                gen_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(gen_features * 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                gen_features * 2,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
        self.embed = nn.Embedding(num_classes, embedding_dim)

    def forward(self, noise, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        noise = torch.cat([noise, embedding], dim=1)
        return self.model(noise)


class Discriminator(nn.Module):
    def __init__(self, channels_img, disc_features, num_classes, img_size):
        super().__init__()
        self.img_size = img_size
        layers = [
            nn.Conv2d(
                channels_img + 1,
                disc_features,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features,
                disc_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(disc_features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 2,
                disc_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(disc_features * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 4, 1, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.Sigmoid(),
        ]
        self.model = nn.Sequential(*layers)
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def forward(self, noise, labels):
        embedding = self.embed(labels).view(
            labels.shape[0], 1, self.img_size, self.img_size
        )
        noise = torch.cat([noise, embedding], dim=1)
        return self.model(noise)

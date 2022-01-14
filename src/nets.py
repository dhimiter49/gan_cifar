import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class Normalizer:
    def __init__(self, normalizers_listalizers):
        self.layers = []
        for normalizer in normalizers_listalizers:
            self.layers.append((getattr(nn, normalizer), normalizer))

    def init(self, input_dim, matrix_dim=None):
        activated_layers = [nn.Identity()]
        for layer, layer_str in self.layers:
            if layer_str == "LayerNorm":
                activated_layers.append(layer([input_dim, matrix_dim, matrix_dim]))
                continue
            activated_layers.append(layer(input_dim))
        return nn.Sequential(*activated_layers)


class DCGAN_Generator(nn.Module):
    def __init__(
        self,
        latent_dim,
        channels_img,
        gen_features,
        num_classes,
        img_size,
        embedding_dim,
        gen_batchnorm,
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
        ]
        if gen_batchnorm == True:
            layers.append(nn.BatchNorm2d(gen_features * 8))
        layers.append(nn.LeakyReLU())
        layers.append(
            nn.ConvTranspose2d(
                gen_features * 8,
                gen_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        if gen_batchnorm == True:
            layers.append(nn.BatchNorm2d(gen_features * 4))
        layers.append(nn.LeakyReLU())
        layers.append(
            nn.ConvTranspose2d(
                gen_features * 4,
                gen_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        if gen_batchnorm == True:
            layers.append(nn.BatchNorm2d(gen_features * 2))
        layers.append(nn.LeakyReLU())
        layers.append(
            nn.ConvTranspose2d(
                gen_features * 2,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self.embed = nn.Embedding(num_classes, embedding_dim)

    def forward(self, noise, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        noise = torch.cat([noise, embedding], dim=1)
        return self.model(noise)


class DCGAN_4x4_Generator(nn.Module):
    def __init__(
        self,
        latent_dim,
        channels_img,
        gen_features,
        num_classes,
        img_size,
        embedding_dim,
        gen_batchnorm,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                latent_dim + int(embedding_dim / (4 * 4)),
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
                kernel_size=7,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
        self.embed = nn.Embedding(num_classes, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, noise, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        embedding = torch.reshape(
            embedding,
            (embedding.shape[0], int(self.embedding_dim / (4 * 4))) + noise.shape[2:],
        )
        noise = torch.cat([noise, embedding], dim=1)
        return self.model(noise)


class DCGAN_4x4_Generator_Conv(nn.Module):
    def __init__(
        self,
        latent_dim,
        channels_img,
        gen_features,
        num_classes,
        img_size,
        embedding_dim,
        gen_batchnorm,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                latent_dim + int(embedding_dim / (4 * 4)),
                gen_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(gen_features * 8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                gen_features * 8,
                gen_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(gen_features * 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                gen_features * 8,
                gen_features * 8,
                kernel_size=4,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(gen_features * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(gen_features * 8, 3, kernel_size=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
        self.embed = nn.Embedding(num_classes, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, noise, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        embedding = torch.reshape(
            embedding,
            (embedding.shape[0], int(self.embedding_dim / (4 * 4))) + noise.shape[2:],
        )
        noise = torch.cat([noise, embedding], dim=1)
        return self.model(noise)


class DCGAN_Discriminator(nn.Module):
    def __init__(
        self,
        channels_img,
        disc_features,
        num_classes,
        img_size,
        disc_batchnorm,
        disc_layernorm,
        disc_instancenorm,
    ):
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


class WGAN_Discriminator(nn.Module):
    def __init__(
        self,
        channels_img,
        disc_features,
        num_classes,
        img_size,
        disc_batchnorm,
        disc_layernorm,
        disc_instancenorm,
    ):
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
            )
        ]
        layers.append(nn.LeakyReLU(0.2))
        layers.append(
            nn.Conv2d(
                disc_features,
                disc_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        if disc_batchnorm == True:
            layers.append(nn.BatchNorm2d(disc_features * 2))
        if disc_layernorm == True:
            layers.append(nn.LayerNorm([disc_features * 2, 8, 8]))
        if disc_instancenorm == True:
            layers.append(nn.InstanceNorm2d(disc_features * 2, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(
            nn.Conv2d(
                disc_features * 2,
                disc_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        if disc_batchnorm == True:
            layers.append(nn.BatchNorm2d(disc_features * 4))
        if disc_layernorm == True:
            layers.append(nn.LayerNorm([disc_features * 4, 4, 4]))
        if disc_instancenorm == True:
            layers.append(nn.InstanceNorm2d(disc_features * 4, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(disc_features * 64, 1))
        self.model = nn.Sequential(*layers)
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def forward(self, noise, labels):
        embedding = self.embed(labels).view(
            labels.shape[0], 1, self.img_size, self.img_size
        )
        noise = torch.cat([noise, embedding], dim=1)
        return self.model(noise)


class WGAN_Spectral_Discriminator(nn.Module):
    def __init__(
        self,
        channels_img,
        disc_features,
        num_classes,
        img_size,
        disc_batchnorm,
        disc_layernorm,
        disc_instancenorm,
    ):
        super().__init__()
        self.img_size = img_size
        layers = [
            spectral_norm(
                nn.Conv2d(
                    channels_img + 1,
                    disc_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
        ]
        layers.append(nn.LeakyReLU(0.2))
        layers.append(
            spectral_norm(
                nn.Conv2d(
                    disc_features,
                    disc_features * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
        )
        if disc_batchnorm == True:
            layers.append(nn.BatchNorm2d(disc_features * 2))
        if disc_layernorm == True:
            layers.append(nn.LayerNorm([disc_features * 2, 8, 8]))
        if disc_instancenorm == True:
            layers.append(nn.InstanceNorm2d(disc_features * 2, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(
            spectral_norm(
                nn.Conv2d(
                    disc_features * 2,
                    disc_features * 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
        )
        if disc_batchnorm == True:
            layers.append(nn.BatchNorm2d(disc_features * 4))
        if disc_layernorm == True:
            layers.append(nn.LayerNorm([disc_features * 4, 4, 4]))
        if disc_instancenorm == True:
            layers.append(nn.InstanceNorm2d(disc_features * 4, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(
            spectral_norm(
                nn.Conv2d(
                    disc_features * 4, 1, kernel_size=4, stride=1, padding=0, bias=False
                )
            )
        )
        self.model = nn.Sequential(*layers)
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def forward(self, noise, labels):
        embedding = self.embed(labels).view(
            labels.shape[0], 1, self.img_size, self.img_size
        )
        noise = torch.cat([noise, embedding], dim=1)
        return self.model(noise)


class DCGAN_Discriminator_FC(nn.Module):
    def __init__(
        self,
        channels_img,
        disc_features,
        num_classes,
        img_size,
        disc_batchnorm,
        disc_layernorm,
        disc_instancenorm,
    ):
        super().__init__()
        self.img_size = img_size
        layers = [
            nn.Conv2d(
                channels_img + 1,
                disc_features * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 4,
                disc_features * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 8,
                disc_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 8,
                disc_features * 16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(disc_features * 64, 1),
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

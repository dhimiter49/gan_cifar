import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgan.layers as gan_layers
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils import weight_norm


class Normalizer:
    def __init__(self, normalizers_list):
        self.layers = []
        for normalizer in normalizers_list:
            if normalizer == "VirtualBatchNorm":
                self.layers.append((getattr(gan_layers, normalizer), normalizer))
                continue
            self.layers.append((getattr(nn, normalizer), normalizer))

    def init(self, input_dim, matrix_dim=None, prob=0.4):
        activated_layers = [nn.Identity()]
        for layer, layer_str in self.layers:
            if layer_str == "LayerNorm":
                activated_layers.append(layer([input_dim, matrix_dim, matrix_dim]))
                continue
            if layer_str == "Dropout":
                activated_layers.append(layer(prob, inplace=True))
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
        normalizers_list,
    ):
        super().__init__()
        normalizer = Normalizer(normalizers_list)
        layers = [
            nn.ConvTranspose2d(
                latent_dim + embedding_dim,
                gen_features * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            normalizer.init(gen_features * 8, 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                gen_features * 8,
                gen_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(gen_features * 4, 8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                gen_features * 4,
                gen_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(gen_features * 2, 16),
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


class DCGAN_4x4_Generator(nn.Module):
    def __init__(
        self,
        latent_dim,
        channels_img,
        gen_features,
        num_classes,
        img_size,
        embedding_dim,
        normalizers_list,
    ):
        super().__init__()
        normalizer = Normalizer(normalizers_list)
        layers = [
            nn.ConvTranspose2d(
                latent_dim + int(embedding_dim / (4 * 4)),
                gen_features * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            normalizer.init(gen_features * 8, 7),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                gen_features * 8,
                gen_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(gen_features * 4, 14),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                gen_features * 4,
                gen_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(gen_features * 2, 28),
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
        normalizers_list,
    ):
        super().__init__()
        normalizer = Normalizer(normalizers_list)
        layers = [
            nn.ConvTranspose2d(
                latent_dim + int(embedding_dim / (4 * 4)),
                gen_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(gen_features * 8, 8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                gen_features * 8,
                gen_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(gen_features * 8, 16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                gen_features * 8,
                gen_features * 8,
                kernel_size=4,
                stride=2,
                padding=0,
                bias=False,
            ),
            normalizer.init(gen_features * 8, 34),
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


class DCGAN_Generator_FC(nn.Module):
    def __init__(
        self,
        latent_dim,
        channels_img,
        gen_features,
        num_classes,
        img_size,
        embedding_dim,
        normalizers_list,
    ):
        super().__init__()
        normalizer = Normalizer(normalizers_list)
        layers = [
            nn.Linear(latent_dim + embedding_dim, 4 * 4 * (gen_features * 8)),
            nn.BatchNorm1d(gen_features * 8 * 4 * 4, 4),
            nn.Unflatten(1, torch.Size([(gen_features * 8), 4, 4])),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                gen_features * 8,
                gen_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(gen_features * 4, 8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                gen_features * 4,
                gen_features * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(gen_features * 2, 16),
            nn.LeakyReLU(),
            weight_norm(
                nn.ConvTranspose2d(
                    gen_features * 2,
                    channels_img,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
        self.embed = nn.Embedding(num_classes, embedding_dim)

    def forward(self, noise, labels):
        embedding = self.embed(labels)
        noise = noise.squeeze()
        noise = torch.cat([noise, embedding], dim=1)
        return self.model(noise)


class DCGAN_Discriminator(nn.Module):
    def __init__(
        self,
        channels_img,
        disc_features,
        num_classes,
        img_size,
        normalizers_list,
    ):
        super().__init__()
        self.img_size = img_size
        normalizer = Normalizer(normalizers_list)
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
            normalizer.init(disc_features * 2, 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 2,
                disc_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(disc_features * 4, 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 4,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        ]
        prediction_layers = [
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*layers)
        self.prediction = nn.Sequential(*prediction_layers)
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def forward(self, noise, labels, feat_matching=False):
        embedding = self.embed(labels).view(
            labels.shape[0], 1, self.img_size, self.img_size
        )
        noise = torch.cat([noise, embedding], dim=1)
        features = self.model(noise)
        if feat_matching:
            return features
        return self.prediction(features)


class DCGAN_Spectral_Discriminator(nn.Module):
    def __init__(
        self,
        channels_img,
        disc_features,
        num_classes,
        img_size,
        normalizers_list,
    ):
        super().__init__()
        self.img_size = img_size
        normalizer = Normalizer(normalizers_list)
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
            ),
            nn.LeakyReLU(0.2),
            spectral_norm(
                nn.Conv2d(
                    disc_features,
                    disc_features * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            normalizer.init(disc_features * 2, 8),
            nn.LeakyReLU(0.2),
            spectral_norm(
                nn.Conv2d(
                    disc_features * 2,
                    disc_features * 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            normalizer.init(disc_features * 4, 4),
            nn.LeakyReLU(0.2),
            spectral_norm(
                nn.Conv2d(
                    disc_features * 4,
                    1,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
        ]

        prediction_layers = [
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*layers)
        self.prediction = nn.Sequential(*prediction_layers)
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def forward(self, noise, labels, feat_matching=False):
        embedding = self.embed(labels).view(
            labels.shape[0], 1, self.img_size, self.img_size
        )
        noise = torch.cat([noise, embedding], dim=1)
        features = self.model(noise)
        if feat_matching:
            return features
        return self.prediction(features)


class WGAN_Discriminator(nn.Module):
    def __init__(
        self,
        channels_img,
        disc_features,
        num_classes,
        img_size,
        normalizers_list,
    ):
        super().__init__()
        self.img_size = img_size
        normalizer = Normalizer(normalizers_list)
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
            normalizer.init(disc_features * 2, 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 2,
                disc_features * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(disc_features * 4, 4),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(disc_features * 64, 1),
        ]
        self.model = nn.Sequential(*layers)
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def forward(self, noise, labels, feat_matching=None):
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
        normalizers_list,
    ):
        super().__init__()
        self.img_size = img_size
        normalizer = Normalizer(normalizers_list)
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
            ),
            nn.LeakyReLU(0.2),
            spectral_norm(
                nn.Conv2d(
                    disc_features,
                    disc_features * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            normalizer.init(disc_features * 2, 8),
            nn.LeakyReLU(0.2),
            spectral_norm(
                nn.Conv2d(
                    disc_features * 2,
                    disc_features * 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            normalizer.init(disc_features * 4, 4),
            nn.LeakyReLU(0.2),
            spectral_norm(
                nn.Conv2d(
                    disc_features * 4, 1, kernel_size=4, stride=1, padding=0, bias=False
                )
            ),
        ]
        self.model = nn.Sequential(*layers)
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def forward(self, noise, labels, feat_matching=None):
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
        normalizers_list,
    ):
        super().__init__()
        self.img_size = img_size
        normalizer = Normalizer(normalizers_list)
        layers = [
            nn.Conv2d(
                channels_img + 1,
                disc_features * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(disc_features * 4, 16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 4,
                disc_features * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(disc_features * 8, 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 8,
                disc_features * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(disc_features * 8, 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 8,
                disc_features * 16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            normalizer.init(disc_features * 16, 2),
        ]

        prediction_layers = [
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(disc_features * 64, 1),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*layers)
        self.prediction = nn.Sequential(*prediction_layers)
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def forward(self, noise, labels, feat_matching=False):
        embedding = self.embed(labels).view(
            labels.shape[0], 1, self.img_size, self.img_size
        )
        noise = torch.cat([noise, embedding], dim=1)
        features = self.model(noise)
        if feat_matching:
            return features
        return self.prediction(features)


class DCGAN_Discriminator_Deeper(nn.Module):
    def __init__(
        self,
        channels_img,
        disc_features,
        num_classes,
        img_size,
        normalizers_list,
    ):
        super().__init__()
        self.img_size = img_size
        normalizer = Normalizer(normalizers_list)
        layers = [
            normalizer.init(channels_img + 1, img_size, 0.2),
            weight_norm(
                nn.Conv2d(
                    channels_img + 1,
                    disc_features * 6,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2),
            weight_norm(
                nn.Conv2d(
                    disc_features * 6,
                    disc_features * 6,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2),
            weight_norm(
                nn.Conv2d(
                    disc_features * 6,
                    disc_features * 6,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2),
            normalizer.init(disc_features * 6, 16, 0.5),
            weight_norm(
                nn.Conv2d(
                    disc_features * 6,
                    disc_features * 12,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2),
            weight_norm(
                nn.Conv2d(
                    disc_features * 12,
                    disc_features * 12,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2),
            weight_norm(
                nn.Conv2d(
                    disc_features * 12,
                    disc_features * 12,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2),
            normalizer.init(disc_features * 12, 8, 0.5),
            weight_norm(
                nn.Conv2d(
                    disc_features * 12,
                    disc_features * 12,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                disc_features * 12,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        ]

        prediction_layers = [
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*layers)
        self.prediction = nn.Sequential(*prediction_layers)
        self.embed = nn.Embedding(num_classes, img_size * img_size)

    def forward(self, noise, labels, feat_matching=False):
        embedding = self.embed(labels).view(
            labels.shape[0], 1, self.img_size, self.img_size
        )
        noise = torch.cat([noise, embedding], dim=1)
        features = self.model(noise)
        if feat_matching:
            return features
        return self.prediction(features)

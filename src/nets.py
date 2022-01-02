import torch
import torch.nn as nn
import torch.nn.functional as F


class DefaultGen(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(3),
            nn.Conv2d(16, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class DefaultDis(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

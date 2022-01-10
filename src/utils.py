import torch
import torch.nn as nn


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
    gradient_norm = gradient.norm(2, dim=1) #l2 norm
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# weight initilization
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


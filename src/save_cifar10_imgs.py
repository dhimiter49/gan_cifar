from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm


working_dir = Path(__file__).parent.parent.absolute()
export_img_dir = working_dir / Path("dataset") / Path("cifar10_iamges")


def main():
    trans = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)]),
        ]
    )

    cifar10_dataset_test = torchvision.datasets.CIFAR10(
        root="./dataset", train=False, download=True, transform=trans
    )
    data_loader_test = torch.utils.data.DataLoader(
        cifar10_dataset_test, batch_size=1000, shuffle=False, num_workers=4
    )

    Path(export_img_dir).mkdir(parents=True, exist_ok=True)
    for i, (data, label) in enumerate(tqdm(data_loader_test)):
        for j, img in enumerate(data):
            img = img / 2 + 0.5
            save_image(img, export_img_dir / Path(str(i * 1000 + j) + ".png"))


if __name__ == "__main__":
    main()

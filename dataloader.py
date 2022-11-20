import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF
from torchvision.transforms import Compose
from config import config


def creat_dataloader(args):
    transform = {
        "train": [
            TF.Resize((args.image_size, args.image_size)),
            TF.RandomHorizontalFlip(),
            TF.RandomRotation(60),
            TF.ToTensor(),
            TF.Normalize(args.mean, args.std)
        ],
        "val": [
            TF.Resize(args.image_size),
            TF.ToTensor(),
            TF.Normalize(args.mean, args.std)
        ]
    }
    dataset = torchvision.datasets.ImageFolder(args.dataset_fold, transform=Compose(transform["train"]))
    val_len = int(len(dataset) * args.test_ratio)
    train_len = len(dataset) - val_len
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])
    # torch.utils.data.Subset.
    # sampler_weight =
    # sampler = torch.utils.data.WeightedRandomSampler()

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=True
    )
    return train_loader, val_loader


if __name__ == '__main__':
    args = config()
    train_loader, _ = creat_dataloader(args)
    for step, (train_x, train_y) in enumerate(train_loader):
        print(train_x.shape)
        print(train_y.shape)
        break
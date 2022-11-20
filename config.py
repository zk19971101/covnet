import argparse


def config():
    parser = argparse.ArgumentParser(description="EfficientNet")
    parser.add_argument("--dataset_fold", default=r"D:\pycharm_projects\flowers")
    parser.add_argument("--test_ratio", default=0.2, type=float)
    parser.add_argument("--val_ratio", default=0.2, type=float)
    parser.add_argument("--image_size", default=320, type=int)
    parser.add_argument("--mean", default=(0.4, 0.4, 0.4))
    parser.add_argument("--std", default=(0.4, 0.4, 0.4))
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--scaler", default=True)
    parser.add_argument("--device", default="cuda")


    args = parser.parse_args()
    return args
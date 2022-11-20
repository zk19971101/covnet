import torch
import torchvision


class Resnet18(torch.nn.Module):
    def __init__(self, classes):
        super(Resnet18, self).__init__()
        self.classes = classes
        self.backbone = torchvision.models.resnet18()
        self.dense1 = torch.nn.Linear(in_features=1000, out_features=128)
        self.dense2 = torch.nn.Linear(in_features=128, out_features=self.classes)

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = self.dense1(x)
        y = self.dense2(x)
        return y


if __name__ == '__main__':
    model = Resnet18(5)
    inputs = torch.randn((2,3,320,320))
    out = model(inputs)
    print(out)

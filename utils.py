import torch
from networks import *
from torch.optim import SGD, Adam


def load_model(args):
    model_list = ["resnet50", "efficnetb3"]
    if args.model_name == model_list[0]:
        model = Resnet50(args.classes).to(args.device)
    elif args.model_name == model_list[1]:
        model = EfficientnetB3(args.classes).to(args.device)
    else:
        model = None
        assert args.model_name in ['resnet50', 'efficnetb3'], \
            f"{args.model_name} not in {model_list}"
    params = [p for p in model.parameters() if p.reqiure_grad]
    optimizer_list = ["SGD", "Adam"]
    if args.optimizer == optimizer_list[0]:
        optimizer = SGD(
            params=params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == optimizer_list[1]:
        optimizer = Adam(
            params=params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = None
        assert args.optimizer in ['SGD', 'Adam'], f"{args.optimizer} not in {optimizer_list}"
    return model, optimizer


def measurement_indictor(y_true: torch.tensor, y_pred: torch.tensor, eps=1e-6):
    '''
    F1 = (precision + recall) / 2 * precision * recall
    precision = TP / (TP + FP) 被预测为真的样本中，正确分类的比例
    recall = TP / (TP + FN) 标签为真的样本中，被正确分类的比例
    t/p 0    1
     0 TP  FN
     1 FP  TN
    kappa = (po - pe) / (1 - pe)
    :param y_true:
    :param y_pred:
    :return:
    '''
    classes = y_pred.shape[-1]
    y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    y_true = y_true.type(torch.int64)
    metric = torch.zeros((classes, classes), dtype=torch.int64)
    metric += torch.bincount(input=y_true*y_pred, minlength=classes**2).reshape(classes, classes)
    acc = metric.diag().sum() / metric.sum()
    precision = metric.diag() / (metric.sum(0) + eps)
    recall = metric.diag() / (metric.sum(1) + eps)
    f1 = (precision + recall) / (2 * precision * recall)
    pe = (metric.sum(0) * metric.sum(1)).sum() / (metric.sum() + eps)
    kappa = ((metric.diag().sum() / metric.sum()) - pe) / (1 - pe +eps)
    return acc, precision, recall, f1, kappa


if __name__ == '__main__':
    y_pred = torch.tensor([[1,2,3],[2,1,3]])
    y_true = torch.tensor([[1,2,3],[2,1,3]])
    print(measurement_indictor(y_true, y_pred))
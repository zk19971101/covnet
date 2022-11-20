import torch


def criterion(args):
    loss_list = ['BCE', 'focal_loss']
    if args.loss == loss_list[0]:
        loss = torch.nn.BCEWithLogitsLoss(weight=args.weight)
    elif args.loss == loss_list[1]:
        loss = FocalLoss(weight=args.weight)
    else:
        loss = None
        assert args.loss in loss_list, f"{args.loss} not in {loss_list}"

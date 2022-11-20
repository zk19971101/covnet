import torch
from config import config
from dataloader import creat_dataloader
from lossfunction import criterion
from utils import load_model
from tqdm import tqdm


def main():
    args = config()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = creat_dataloader(args)
    model, optimizer = load_model(args)
    scaler = torch.cuda.amp.grad_scaler if args.scaler else None
    lossfunction = criterion(args)


def train(args, train_loader, val_loader, model, optimizer, criterion, scaler):
    model = model.to(args.device)
    for epoch in range(1, args.epochs+1):
        model.train()
        for step, (train_x, train_y) in enumerate(tqdm(train_loader)):
            train_x, train_y = train_x.to(args.device), train_y.to(args.device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                res = model(train_x)
                loss = criterion(inputs=res.type(torch.float32), targets=train_y.type(torch.float32))
            optimizer.zero_grad()
            scaler.apply(optimizer)
            scaler.update()
        torch.cuda.empty_cache()

        model.eval()
        for val_step, (val_x, val_y) in enumerate(val_loader):
            val_x, val_y = val_x.to(args.device), val_y.to(args.device)
            val_res = model(val_x)
            val_loss = criterion(inputs=val_res, targets=val_y)

        torch.cuda.empty_cache()

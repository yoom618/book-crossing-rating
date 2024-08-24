import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
from torch.optim import SGD, Adam


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


def train(args, model, dataloader, logger, setting):
    minimum_loss = 999999999

    loss_fns = {'MSE': MSELoss(), 'RMSE': RMSELoss(), 'MAE': L1Loss()}
    if args.loss_fn in loss_fns:
        loss_fn = loss_fns[args.loss_fn]
    else:
        assert False, f'{args.loss_fn}을/를 loss_fn에 추가해주세요.'

    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'ADAM':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        assert False, f'{args.optimizer} 옵티마이저는 현재 구현되어 있지 않습니다.'

    if args.lr_scheduler == '':
        lr_scheduler = None
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    elif args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        assert False, f'{args.lr_scheduler} 스케쥴러는 현재 구현되어 있지 않습니다.'

    for epoch in tqdm(range(args.epochs)):
        model.train()
        total_loss, train_len = 0, len(dataloader['train_dataloader'])

        for data in dataloader['train_dataloader']:
            if args.model == 'CNN_FM':
                x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
            elif args.model == 'DeepCoNN':
                x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
            else:
                x, y = data[0].to(args.device), data[1].to(args.device)
            y_hat = model(x)
            loss = loss_fn(y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        msg = f'[Epoch {epoch+1:02d}/{args.epochs:02d}]'
        train_loss = total_loss / train_len
        msg += f'\nTrain Loss: {train_loss:.3f}'
        if args.test_size != 0:
            valid_loss = valid(args, model, dataloader['valid_dataloader'], loss_fn)
            msg += f'\nValid Loss: {valid_loss:.3f}'
            
            valid_metrics = dict()
            for metric in args.metric:
                valid_metric = valid(args, model, dataloader['valid_dataloader'], loss_fns[metric])
                valid_metrics[metric] = valid_metric
            for metric, value in valid_metrics.items():
                msg += f' | {metric}: {value:.3f}'
            print(msg)
            logger.log(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss, valid_metrics=valid_metrics)
        else:
            print(msg)
            logger.log(epoch=epoch+1, train_loss=train_loss)
        
        best_loss = valid_loss if args.test_size != 0 else train_loss
        if minimum_loss > best_loss:
            minimum_loss = best_loss
            os.makedirs(args.saved_model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{args.saved_model_path}/{setting.save_time}_{args.model}_model.pt')
    
    logger.close()
    
    return model


def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0

    for data in dataloader:
        if args.model == 'CNN_FM':
            x, y = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, y = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        batch +=1
        
    return total_loss/batch


def test(args, model, dataloader, setting):
    predicts = list()
    if args.use_best_model == True:
        model.load_state_dict(torch.load(f'./saved_models/{setting.save_time}_{args.model}_model.pt', weights_only=False))
    else:
        pass
    model.eval()

    for idx, data in enumerate(dataloader['test_dataloader']):
        if args.model == 'CNN_FM':
            x, _ = [data['user_isbn_vector'].to(args.device), data['img_vector'].to(args.device)], data['label'].to(args.device)
        elif args.model == 'DeepCoNN':
            x, _ = [data['user_isbn_vector'].to(args.device), data['user_summary_merge_vector'].to(args.device), data['item_summary_vector'].to(args.device)], data['label'].to(args.device)
        else:
            x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts

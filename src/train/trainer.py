import os
from tqdm import tqdm
import torch
from src.loss import loss as loss_module
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module


METRIC_NAMES = {
    'RMSELoss': 'RMSE',
    'MSELoss': 'MSE',
    'MAELoss': 'MAE'
}

def train(args, model, dataloader, logger, setting):

    if args.wandb:
        import wandb
    
    minimum_loss = 999999999

    loss_fn = getattr(loss_module, args.loss)().to(args.device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(optimizer_module, args.optimizer.type)(trainable_params,
                                                               **args.optimizer.args)

    if args.lr_scheduler.use:
        lr_scheduler = getattr(scheduler_module, args.lr_scheduler.type)(optimizer, 
                                                                         **args.lr_scheduler.args)
    else:
        lr_scheduler = None

    for epoch in tqdm(range(args.train.epochs)):
        model.train()
        total_loss, train_len = 0, len(dataloader['train_dataloader'])

        for data in dataloader['train_dataloader']:
            if args.model_args[args.model].datatype == 'image':
                x, y = [data['user_book_vector'].to(args.device), data['img_vector'].to(args.device)], data['rating'].to(args.device)
            elif args.model_args[args.model].datatype == 'text':
                x, y = [data['user_book_vector'].to(args.device), data['user_summary_vector'].to(args.device), data['book_summary_vector'].to(args.device)], data['rating'].to(args.device)
            else:
                x, y = data[0].to(args.device), data[1].to(args.device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if args.lr_scheduler.use:
            lr_scheduler.step()
        
        msg = f'[Epoch {epoch+1:02d}/{args.train.epochs:02d}]'
        train_loss = total_loss / train_len
        msg += f'\nTrain Loss: {train_loss:.3f}'
        if args.dataset.valid_ratio != 0:  # valid 데이터가 존재할 경우
            valid_loss = valid(args, model, dataloader['valid_dataloader'], loss_fn)
            msg += f'\nValid Loss: {valid_loss:.3f}'
            
            valid_metrics = dict()
            for metric in args.metrics:
                metric_fn = getattr(loss_module, metric)().to(args.device)
                valid_metric = valid(args, model, dataloader['valid_dataloader'], metric_fn)
                valid_metrics[METRIC_NAMES[metric]] = valid_metric
            for metric, value in valid_metrics.items():
                msg += f' | {metric}: {value:.3f}'
            print(msg)
            logger.log(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss, valid_metrics=valid_metrics)
            if args.wandb:
                wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss, **valid_metrics})
        else:  # valid 데이터가 없을 경우
            print(msg)
            logger.log(epoch=epoch+1, train_loss=train_loss)
            if args.wandb:
                wandb.log({'train_loss': train_loss})
        
        if args.train.save_best_model:
            best_loss = valid_loss if args.dataset.valid_ratio != 0 else train_loss
            if minimum_loss > best_loss:
                minimum_loss = best_loss
                os.makedirs(args.train.save_dir.checkpoint, exist_ok=True)
                torch.save(model.state_dict(), f'{args.train.save_dir.checkpoint}/{setting.save_time}_{args.model}_best.pt')
        else:
            os.makedirs(args.train.save_dir.checkpoint, exist_ok=True)
            torch.save(model.state_dict(), f'{args.train.save_dir.checkpoint}/{setting.save_time}_{args.model}_e{epoch}.pt')
    
    logger.close()
    
    return model


def valid(args, model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0

    for data in dataloader:
        if args.model_args[args.model].datatype == 'image':
            x, y = [data['user_book_vector'].to(args.device), data['img_vector'].to(args.device)], data['rating'].to(args.device)
        elif args.model_args[args.model].datatype == 'text':
            x, y = [data['user_book_vector'].to(args.device), data['user_summary_vector'].to(args.device), data['book_summary_vector'].to(args.device)], data['rating'].to(args.device)
        else:
            x, y = data[0].to(args.device), data[1].to(args.device)
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        batch +=1
        
    return total_loss/batch


def test(args, model, dataloader, setting, checkpoint=None):
    predicts = list()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        if args.train.save_best_model:
            model_path = f'{args.train.save_dir.checkpoint}/{setting.save_time}_{args.model}_best.pt'
        else:
            # best가 아닐 경우 마지막 에폭으로 테스트하도록 함
            model_path = f'{args.train.save_dir.checkpoint}/{setting.save_time}_{args.model}_e{args.train.epochs}.pt'
        model.load_state_dict(torch.load(model_path, weights_only=True))
    
    model.eval()
    for data in dataloader['test_dataloader']:
        if args.model_args[args.model].datatype == 'image':
            x = [data['user_book_vector'].to(args.device), data['img_vector'].to(args.device)]
        elif args.model_args[args.model].datatype == 'text':
            x = [data['user_book_vector'].to(args.device), data['user_summary_vector'].to(args.device), data['book_summary_vector'].to(args.device)]
        else:
            x = data[0].to(args.device)
        y_hat = model(x)
        predicts.extend(y_hat.tolist())
    return predicts

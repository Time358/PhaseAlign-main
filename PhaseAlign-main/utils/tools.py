import os

import torch
from tqdm import tqdm

PROGRESS_KWARGS = dict(ncols=40, bar_format='{l_bar}{bar:10}{percentage:3.0f}% {n_fmt}/{total_fmt}',
                       mininterval=1.0, leave=False)


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7):
        self.accelerator = accelerator
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, val_loss, model, path, epoch=None):
        score = -val_loss
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model, path)
            return
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, model, path):
        if self.accelerator is not None:
            if not self.accelerator.is_main_process:
                return
            model = self.accelerator.unwrap_model(model)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint'))


def vali(args, accelerator, model, vali_loader, criterion, mae_metric, show=False):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n_samples = 0
    f_dim = -1 if args.features == 'MS' else 0
    show = show and (accelerator is None or accelerator.is_main_process)
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
                vali_loader, disable=not show, **PROGRESS_KWARGS):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            outputs = model(batch_x, batch_x_mark, None, batch_y_mark)

            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]
            if outputs.shape != batch_y.shape:
                raise RuntimeError(f'prediction shape {tuple(outputs.shape)} != '
                                   f'target shape {tuple(batch_y.shape)}')

            pred, true = accelerator.gather_for_metrics((outputs, batch_y))
            n = true.shape[0]
            mse_sum += criterion(pred, true).item() * n
            mae_sum += mae_metric(pred, true).item() * n
            n_samples += n

    model.train()
    if n_samples == 0:
        raise RuntimeError('empty validation loader')
    return mse_sum / n_samples, mae_sum / n_samples

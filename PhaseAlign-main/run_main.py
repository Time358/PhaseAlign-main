import argparse
import gc
import json
import os
import random
import sys
import time

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import nn, optim
from tqdm import tqdm

from data_provider.data_factory import data_provider
from models import PhaseAlign
from utils.metrics import metric
from utils.tools import PROGRESS_KWARGS, EarlyStopping, vali

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


def spectral_phase_loss(preds, targets, eps=1e-8):
    pred_spec = torch.fft.rfft(preds.float(), dim=1)
    true_spec = torch.fft.rfft(targets.float(), dim=1)
    amp = true_spec.abs()
    amp = amp.clone()
    amp[:, 0, :] = 0.0
    weight = amp / amp.sum(dim=1, keepdim=True).clamp_min(eps)
    phase_gap = torch.angle(true_spec) - torch.angle(pred_spec)
    per_channel = (weight * (1.0 - torch.cos(phase_gap))).sum(dim=1)
    return per_channel.mean()


def alignment_loss(module, history_x, future_x, history_mark):
    hist_repr, future_repr = module.alignment_repr(history_x, future_x, history_mark)
    module.clear_align_cache()
    return (1.0 - torch.cosine_similarity(hist_repr, future_repr, dim=-1)).mean()


@torch.no_grad()
def collect_predictions(args, accelerator, model, loader):
    model.eval()
    preds, trues = [], []
    f_dim = -1 if args.features == 'MS' else 0
    for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(
            loader, disable=not args.show_progress, **PROGRESS_KWARGS):
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

        outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
        preds.append(outputs.detach().float().cpu().numpy())
        trues.append(batch_y.detach().float().cpu().numpy())

    if not preds:
        raise RuntimeError('no evaluation batches were produced')
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def report_metrics(accelerator, tag, preds, trues):
    mae, mse, rmse = metric(preds, trues)
    accelerator.print(f'{tag}MSE: {mse:.7f} MAE: {mae:.7f} RMSE: {rmse:.7f}')
    return {'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse)}


def build_parser():
    p = argparse.ArgumentParser(description='PhaseAlign')

    p.add_argument('--task_name', type=str, default='long_term_forecast')
    p.add_argument('--is_training', type=int, default=1, choices=[0, 1])
    p.add_argument('--model_id', type=str, required=True)
    p.add_argument('--model_comment', type=str, default='none')
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--seeds', type=str, default='')

    p.add_argument('--data', type=str, required=True)
    p.add_argument('--root_path', type=str, default='./dataset')
    p.add_argument('--data_path', type=str, default='ETTh1.csv')
    p.add_argument('--features', type=str, default='M', choices=['M', 'S', 'MS'])
    p.add_argument('--target', type=str, default='OT')
    p.add_argument('--freq', type=str, default='h')
    p.add_argument('--checkpoints', type=str, default='./checkpoints/')

    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--enc_in', type=int, default=7)
    p.add_argument('--d_model', type=int, default=384)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--e_layers', type=int, default=4)
    p.add_argument('--d_ff', type=int, default=768)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--patch_len', type=int, default=32)
    p.add_argument('--stride', type=int, default=16)
    p.add_argument('--top_k', type=int, default=9)
    p.add_argument('--fourier_fmax', type=float, default=100.0)

    p.add_argument('--phase_weight', type=float, default=0.1)
    p.add_argument('--align_weight', type=float, default=None)
    p.add_argument('--phase_eps', type=float, default=1e-8)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--eval_batch_size', type=int, default=8)
    p.add_argument('--train_epochs', type=int, default=10)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--show_progress', type=int, default=1, choices=[0, 1])
    return p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_setting(args, seed):
    return '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_P{}_S{}_K{}_sd{}_{}'.format(
        args.task_name, args.model_id, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
        args.e_layers, args.d_ff, args.patch_len, args.stride, args.top_k,
        seed, args.model_comment)


def run_experiment(args, accelerator, seed):
    set_seed(seed)

    train_data, train_loader = data_provider(args, 'train', seed)
    vali_data, vali_loader = data_provider(args, 'val', seed)
    test_data, test_loader = data_provider(args, 'test', seed)
    if train_data.enc_in != args.enc_in:
        raise ValueError(f'--enc_in {args.enc_in} does not match the {train_data.enc_in} '
                         f'variables found in {args.data_path}')

    model = PhaseAlign(args).float()
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    n_patch = model.n_patch
    f_dim = -1 if args.features == 'MS' else 0

    setting = build_setting(args, seed)
    path = os.path.join(args.checkpoints, f'{setting}')
    if accelerator.is_main_process:
        os.makedirs(path, exist_ok=True)

        meta = {
            'argv': sys.argv,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'seed': seed,
            'torch': torch.__version__,
            'args': {k: v for k, v in vars(args).items() if k != 'seed_list'},
            'effective': {
                'n_trainable_params': n_params,
                'n_patches': n_patch,
                'train_samples': len(train_data),
                'val_samples': len(vali_data),
                'test_samples': len(test_data),
                'train_batch_size': args.batch_size,
                'eval_batch_size': args.eval_batch_size,
                'train_epochs': args.train_epochs,
                'patience': args.patience,
                'base_learning_rate': float(args.learning_rate),
                'lambda_phase': args.phase_weight,
                'lambda_align': args.align_weight,
            },
        }
        with open(os.path.join(path, 'run_meta.json'), 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2)

    model_optim = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    if args.is_training == 0:
        test_loader, model = accelerator.prepare(test_loader, model)
    else:
        train_loader, vali_loader, test_loader, model, model_optim = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim)
    accelerator.print(f'[seed {seed}] device {next(model.parameters()).device} '
                      f'params {n_params:,} patches {n_patch}')

    def load_checkpoint():
        ckpt = os.path.join(path, 'checkpoint')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'no checkpoint at {ckpt}; run with --is_training 1 first')
        state = torch.load(ckpt, map_location=accelerator.device)
        accelerator.unwrap_model(model).load_state_dict(state)
        accelerator.print(f'[seed {seed}] loaded {ckpt}')

    if args.is_training == 0:
        load_checkpoint()
        stats = report_metrics(accelerator, 'Test ',
                               *collect_predictions(args, accelerator, model, test_loader))
        accelerator.wait_for_everyone()
        return {'seed': seed, 'path': path, 'metrics': stats, 'n_params': n_params}

    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)
    train_steps = len(train_loader)
    accelerator.print(f'[seed {seed}] lambda_p={args.phase_weight} lambda_a={args.align_weight}')

    for epoch in range(args.train_epochs):
        model.train()
        epoch_start = time.time()
        train_loss = []
        progress = tqdm(train_loader, total=train_steps,
                        desc=f'Epoch {epoch + 1}/{args.train_epochs}',
                        disable=not (accelerator.is_main_process and args.show_progress),
                        **PROGRESS_KWARGS)
        for batch_x, batch_y, batch_x_mark, batch_y_mark in progress:
            model_optim.zero_grad()
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            loss = criterion(outputs, batch_y)
            if args.phase_weight:
                loss = loss + args.phase_weight * spectral_phase_loss(outputs, batch_y, args.phase_eps)
            if args.align_weight:
                loss = loss + args.align_weight * alignment_loss(
                    accelerator.unwrap_model(model), batch_x, batch_y, batch_x_mark)

            accelerator.backward(loss)
            model_optim.step()
            train_loss.append(loss.item())
            progress.set_postfix(loss=f'{loss.item():.5f}', refresh=False)

        train_loss = float(np.mean(train_loss))
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_loader, criterion, mae_metric,
                                        show=bool(args.show_progress))
        accelerator.print(f'Epoch {epoch + 1} | {time.time() - epoch_start:.1f}s | '
                          f'Train {train_loss:.7f} | Vali MSE {vali_loss:.7f} | '
                          f'Vali MAE {vali_mae_loss:.7f} | lr {model_optim.param_groups[0]["lr"]:.2e}')

        early_stopping(vali_mae_loss, model, path, epoch=epoch + 1)
        if early_stopping.early_stop:
            accelerator.print(f'Early stopping at epoch {epoch + 1} '
                              f'(best epoch {early_stopping.best_epoch})')
            break

    accelerator.wait_for_everyone()
    load_checkpoint()
    stats = report_metrics(accelerator, 'Test ',
                           *collect_predictions(args, accelerator, model, test_loader))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        with open(os.path.join(path, 'metrics.json'), 'w', encoding='utf-8') as fh:
            json.dump({'seed': seed, 'test': stats, 'n_trainable_params': n_params,
                       'best_epoch': early_stopping.best_epoch}, fh, indent=2)
    return {'seed': seed, 'path': path, 'metrics': stats, 'n_params': n_params,
            'best_epoch': early_stopping.best_epoch}


def summarize(results, accelerator):
    if len(results) == 1:
        return None
    keys = ('mse', 'mae')
    lines = ['[summary] mean +- std over seeds']
    summary = {'seeds': [r['seed'] for r in results],
               'per_seed': [{'seed': r['seed'], **r['metrics']} for r in results]}
    for k in keys:
        vals = np.array([r['metrics'][k] for r in results], dtype=np.float64)
        summary[k] = {'mean': float(vals.mean()), 'std': float(vals.std(ddof=1)) if len(vals) > 1
                      else 0.0, 'values': vals.tolist()}
        lines.append(f'  {k.upper()}: {vals.mean():.4f} +- {summary[k]["std"]:.4f} '
                     f'(n={len(vals)})')
    accelerator.print('\n'.join(lines))
    return summary


def main():
    args = build_parser().parse_args()
    if args.align_weight is None:
        args.align_weight = args.phase_weight
    args.seed_list = [int(s) for s in args.seeds.split(',') if s.strip()] or [args.seed]

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    results = []
    for seed in args.seed_list:
        results.append(run_experiment(args, accelerator, seed))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(results, accelerator)
    if summary is not None and accelerator.is_main_process:
        tag = f'{args.model_id}-{args.model_comment}'
        with open(os.path.join(args.checkpoints, f'{tag}_seed_summary.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2)
    accelerator.print('done')


if __name__ == '__main__':
    main()

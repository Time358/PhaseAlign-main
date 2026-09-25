
import argparse
import sys
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate import DistributedDataParallelKwargs

from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import PhaseAlignModel

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import os
os.environ["DS_SKIP_MPI_INIT"] = "1"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import EarlyStopping, adjust_learning_rate, vali
import utils.tools as tools_mod
from utils.metrics import metric

# 精简进度条格式：去掉 ETA/速度，缩短条宽，降低刷新频率
SHORT_BAR_KWARGS = dict(ncols=40,
                        bar_format='{l_bar}{bar:10}{percentage:3.0f}% {n_fmt}/{total_fmt}',
                        mininterval=1.0, leave=False)


def seasonal_residual_preds(batch_x, pred_len, period):
    """Naive seasonal copy baseline.

    For each future horizon h (0-indexed, h = 0..pred_len-1), copy the value
    from `period` steps earlier, wrapped by phase:
        y_hat^seasonal[h] = batch_x[:, L - period + (h % period), :]
    Uses the last `period` samples of the input window, so it only requires the
    history (no future leakage) and is valid for any pred_len via phase wrap.
    batch_x: [B, L, C] -> returns [B, pred_len, C].
    """
    period = max(1, int(period))
    season = batch_x[:, -period:, :]                                    # [B, period, C]
    idx = torch.arange(pred_len, device=batch_x.device) % period        # [pred_len]
    return season[:, idx, :]                                            # [B, pred_len, C]


def seasonal_prototype_preds(batch_x, pred_len, period, weights):
    """Weighted multi-day seasonal prototype.

    For future horizon h (phase r = h % period) and day offset d (0 = most
    recent cycle), the history value at that phase is
        x[:, L - (d+1)*period + r, :]
    The prototype prediction is
        y_proto[h] = sum_d w_d * x[:, L - (d+1)*period + r, :]
    averaged over the last len(weights) cycles. If fewer full cycles exist in
    the window than len(weights), the weights are renormalized over the
    available offsets. weights=[] falls back to the naive single-day copy.
    This reduces the risk that one anomalous day corrupts the whole copy.
    """
    period = max(1, int(period))
    device, dtype = batch_x.device, batch_x.dtype
    n_avail = batch_x.shape[1] // period
    n_off = min(len(weights), n_avail)
    if n_off < 1:
        return seasonal_residual_preds(batch_x, pred_len, period)

    w = torch.tensor(weights[:n_off], dtype=dtype, device=device)
    w = w / w.sum()
    idx = torch.arange(pred_len, device=device) % period
    out = torch.zeros(batch_x.shape[0], pred_len, batch_x.shape[2], dtype=dtype, device=device)
    L = batch_x.shape[1]
    for d in range(n_off):
        seg = batch_x[:, L - (d + 1) * period: L - d * period, :]       # [B, period, C]
        out += w[d] * seg[:, idx, :]
    return out


def cand_label(args):
    """Human-readable name of the post-hoc candidate."""
    if getattr(args, 'seasonal_weights', None):
        return 'proto' + ','.join(f'{w:g}' for w in args.seasonal_weights)
    return 'naive'


def posthoc_cand(batch_x, args, f_dim):
    """Post-hoc reference predictor aligned to [B, pred_len, C_out].
    Uses the weighted multi-day prototype when --seasonal_weights is set,
    otherwise the naive single-day seasonal copy."""
    if getattr(args, 'seasonal_weights', None):
        cand = seasonal_prototype_preds(batch_x, args.pred_len, args.seasonal_period, args.seasonal_weights)
    else:
        cand = seasonal_residual_preds(batch_x, args.pred_len, args.seasonal_period)
    return cand[:, -args.pred_len:, f_dim:]


def posthoc_report(base_preds, cand_preds, trues, alphas, name='naive'):
    """Sweep a in Y = Ya + a*(Ycand - Ya) = (1-a)*Ya + a*Ycand."""
    print('=' * 70)
    print(f'Post-hoc sweep [{name}]:  Y = Ya + a*(Ycand - Ya)')
    results = []
    for a in alphas:
        blended = (1.0 - a) * base_preds + a * cand_preds
        mae, mse, _rmse, _mape, _mspe = metric(blended, trues)
        results.append((a, mae, mse))
        print(f'  alpha={a:.3f}   MAE={mae:.7f}   MSE={mse:.7f}')
    best_a, best_mae, best_mse = min(results, key=lambda r: r[1])
    base_mae, base_mse, *_ = metric(base_preds, trues)
    print(f'  best  alpha={best_a:.3f}   MAE={best_mae:.7f}   MSE={best_mse:.7f}')
    print(f'  vs base MAE={base_mae:.7f} -> delta={best_mae - base_mae:+.7f} '
          f'({"IMPROVED" if best_mae < base_mae else "no gain"})')
    print('=' * 70)
    return results


def vali_fold(args, accelerator, model, loader, alpha):
    """Run one pass over `loader` and return (mse, mae) of the naive-24h-residual
    folded prediction Y = Ya + alpha*(Yseasonal - Ya) = (1-alpha)*Ya + alpha*Yseasonal.
    Used so each training epoch directly reports the final corrected metric,
    instead of a separate post-hoc sweep at the end."""
    mse_all, mae_all = [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(loader, disable=not args.show_progress, **SHORT_BAR_KWARGS):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()

            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            cand = posthoc_cand(batch_x, args, f_dim)
            outputs = (1.0 - alpha) * outputs + alpha * cand

            pred = accelerator.gather_for_metrics(outputs.detach())
            true = accelerator.gather_for_metrics(batch_y.detach())
            mse_all.append(nn.MSELoss()(pred, true).item())
            mae_all.append(nn.L1Loss()(pred, true).item())
    model.train()
    return float(np.average(mse_all)), float(np.average(mae_all))


parser = argparse.ArgumentParser(description='PhaseAlign-LLM')

fix_seed = 2026
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='PhaseAlign',
                    help='model name, options: [PhaseAlign]')
parser.add_argument('--seed', type=int, default=2026, help='random seed')
parser.add_argument('--seed_offset', type=int, default=0,
                    help='added to --seed for all RNGs (torch/np/python); use to '
                         'de-correlate sensitivity grids sharing one seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=96, help='patch length')
parser.add_argument('--stride', type=int, default=48, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--phase_model', type=str, default='PhaseAlign', help='core model name')
parser.add_argument('--phase_dim', type=int, default=16, help='core model dimension')
parser.add_argument('--use_time_mark', type=int, default=1,
                    help='inject timestamp marks into encoder/decoder branches (0 to disable)')
parser.add_argument('--use_rel_phase', type=int, default=1,
                    help='keep encoder relative phase (RoPE + phase bias), 0 to ablate')
parser.add_argument('--use_fut_phase', type=int, default=1,
                    help='keep decoder future absolute-phase extrapolation, 0 to ablate')
parser.add_argument('--use_tf_interaction', type=int, default=1,
                    help='keep time-frequency cross-attention fusion, 0 to ablate')
parser.add_argument('--shared_phase', type=int, default=0, choices=[0, 1, 2],
                    help='stage-specificity control: 0=stage-specific, 1=shared relative operator, '
                         '2=shared absolute/window-referenced operator')
parser.add_argument('--use_seasonal_residual', type=int, default=0,
                    help='add naive 24h seasonal residual at evaluation time')
parser.add_argument('--seasonal_period', type=int, default=24,
                    help='seasonal copy period for residual baseline')
parser.add_argument('--seasonal_weights', type=str, default='',
                    help='multi-day prototype weights, e.g. "0.5,0.3,0.2"; empty = naive '
                         'single-day copy (w1 only)')
parser.add_argument('--seasonal_alphas', type=str, default='0.05,0.1,0.2,0.3',
                    help='comma-separated alpha values for seasonal residual sweep')
parser.add_argument('--seasonal_alpha', type=float, default=-1,
                    help='if >=0, fold naive 24h residual into the model output with this fixed '
                         'alpha and report ONE final result (disables the alpha sweep)')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--align_epochs', type=int, default=50, help='alignment epochs')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--early_stop', type=int, default=1,
                    help='enable early stopping during training (1=on, 0=off)')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--mse_weight', type=float, default=0.5, help='weight for MSE in mixed loss')
parser.add_argument('--mae_weight', type=float, default=0.5, help='weight for MAE in mixed loss')
parser.add_argument('--phase_weight', type=float, default=0.1, help='weight for phase alignment loss')
parser.add_argument('--align_weight', type=float, default=None,
                    help='weight for masked future alignment loss (default: same as phase_weight)')
parser.add_argument('--top_k', type=int, default=9, help='number of top frequency bins (PhaseAlign top_k)')
parser.add_argument('--fourier_fmax', type=float, default=100.0, help='fixed Fourier basis max freq')
parser.add_argument('--use_fixed_basis', type=int, default=1, choices=[0, 1],
                    help='1=keep fixed Fourier basis, 0=zero it (keep adaptive only)')
parser.add_argument('--phase_eps', type=float, default=1e-8, help='numerical stability epsilon for phase loss')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--show_progress', type=int, default=1,
                    help='show tqdm progress bars (1=show, 0=hide)')
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()
if getattr(args, 'align_weight', None) is None:
    args.align_weight = args.phase_weight
# Per-run seed: every sensitivity grid gets its own RNG stream so grids that
# share all other settings cannot produce bit-identical trajectories.
fix_seed = int(args.seed) + int(args.seed_offset)
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(fix_seed)
args.seasonal_alphas = [float(a) for a in args.seasonal_alphas.split(',') if a.strip()]
args.seasonal_weights = [float(x) for x in args.seasonal_weights.split(',') if x.strip()]
# 统一开关：关掉后隐藏所有验证/测试进度条
tools_mod.VALI_SHOW_PROGRESS = bool(args.show_progress)
# static_graph=True：训练中会进行两次前向（主预测 + future_alignment_repr），
# 共享参数（future_align_proj）在同一个 iteration 会被使用多次，需启用静态图模式
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)
# 分布式后端完全由 Accelerate 配置文件控制（你描述的工作流）：
#   1) accelerate config 生成配置文件，把后端选为 DeepSpeed；
#   2) accelerate launch --config_file <cfg.yaml> 启动；
#   3) Accelerate 在后台自动创建 DeepSpeed 引擎，ZeRO 等优化自动生效。
# 代码里不再显式创建 DeepSpeedPlugin，避免与配置文件方式冲突。
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
if args.use_amp and accelerator.distributed_type == DistributedType.DEEPSPEED:
    # DeepSpeed 通过 ds_config 的 bf16 auto_cast 统一管理精度，避免与手动 AMP 冲突
    accelerator.print('[run_main] DeepSpeed backend detected: precision is handled by the '
                      'DeepSpeed config, disabling --use_amp.')
    args.use_amp = 0

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    if args.model == 'PhaseAlign':
        model = PhaseAlignModel(args).float()
    else:
        raise ValueError(f'Unsupported model: {args.model}')

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    # --early_stop 0: disable early termination (but still save best checkpoint)
    es_patience = args.patience if args.early_stop else max(args.patience, args.train_epochs)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=es_patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    if args.loss.upper() == 'MIXED':
        loss_mse = nn.MSELoss()
        loss_mae = nn.L1Loss()

    def phase_alignment_loss(preds, targets):
        preds_fft = torch.fft.rfft(preds.float(), dim=1)
        targets_fft = torch.fft.rfft(targets.float(), dim=1)
        phase_diff = torch.angle(targets_fft) - torch.angle(preds_fft)
        return (1.0 - torch.cos(phase_diff)).mean()

    def masked_future_alignment_loss(history_x, future_x, future_mask):
        future_model = model.module if hasattr(model, 'module') else model
        hist_repr, future_repr = future_model.future_alignment_repr(history_x, future_x, future_mask)
        return (1.0 - torch.cosine_similarity(hist_repr, future_repr, dim=-1)).mean()

    if args.is_training == 0:
        test_loader, model = accelerator.prepare(test_loader, model)
    else:
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    accelerator.print("model device:", next(model.parameters()).device)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    def evaluate_only():
        best_ckpt = os.path.join(path, 'checkpoint')
        if not os.path.exists(best_ckpt):
            print(f'Best checkpoint not found at {best_ckpt}')
            return

        print(f'Loading best checkpoint from {best_ckpt}')
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = torch.load(best_ckpt, map_location=accelerator.device)
        unwrapped_model.load_state_dict(state_dict)

        fold_test = args.use_seasonal_residual and args.seasonal_alpha >= 0
        if fold_test:
            test_loss = test_mae_loss = None
        else:
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        model.eval()
        preds, trues, seasonals = [], [], []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), disable=not args.show_progress, **SHORT_BAR_KWARGS):
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float().to(accelerator.device)
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]

                if args.use_seasonal_residual:
                    cand = posthoc_cand(batch_x, args, f_dim)
                    if args.seasonal_alpha >= 0:
                        outputs = (1.0 - args.seasonal_alpha) * outputs + args.seasonal_alpha * cand
                        outs_g, tru_g = accelerator.gather_for_metrics((outputs, batch_y))
                    else:
                        outs_g = accelerator.gather_for_metrics((outputs, cand, batch_y))
                        outs_g, cand_g, tru_g = outs_g
                        seasonals.append(cand_g.detach().float().cpu().numpy())
                else:
                    outs_g, tru_g = accelerator.gather_for_metrics((outputs, batch_y))
                preds.append(outs_g.detach().float().cpu().numpy())
                trues.append(tru_g.detach().float().cpu().numpy())

        if len(preds) > 0:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            if fold_test:
                print(f'MAE[{cand_label(args)} a={args.seasonal_alpha:.2f}]: {mae:.7f} '
                      f'MSE[{cand_label(args)} a={args.seasonal_alpha:.2f}]: {mse:.7f} RMSE: {rmse:.7f}')
            else:
                print(f'Test Loss: {test_loss:.7f} Test MAE Loss: {test_mae_loss:.7f}')
                print(f'MAE: {mae:.7f} MSE: {mse:.7f} RMSE: {rmse:.7f} MAPE: {mape:.7f} MSPE: {mspe:.7f}')
                if args.use_seasonal_residual and args.seasonal_alpha < 0 and len(seasonals) > 0:
                    seasonals = np.concatenate(seasonals, axis=0)
                    posthoc_report(preds, seasonals, trues, args.seasonal_alphas, name=cand_label(args))
        else:
            print(f'Test Loss: {test_loss:.7f} Test MAE Loss: {test_mae_loss:.7f}')
            print('No test samples were collected for metric calculation.')

    if args.is_training == 0:
        evaluate_only()
        sys.exit(0)

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        data_time = 0.0
        transfer_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optim_time = 0.0
        last_iter_end = time.time()
        train_progress = tqdm(
            enumerate(train_loader),
            total=train_steps,
            desc=f"Epoch {epoch + 1}/{args.train_epochs}",
            disable=not (accelerator.is_local_main_process and args.show_progress),
            **SHORT_BAR_KWARGS,
        )
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in train_progress:
            data_loaded_time = time.time()
            data_time += data_loaded_time - last_iter_end
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            if accelerator.device.type == 'cuda':
                torch.cuda.synchronize()
            transfer_time += time.time() - data_loaded_time
            forward_start = time.time()

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    pred_loss = criterion(outputs, batch_y)
                    if args.loss.upper() == 'MIXED':
                        pred_loss = args.mse_weight * loss_mse(outputs, batch_y) + args.mae_weight * loss_mae(outputs, batch_y)
                    phase_loss = phase_alignment_loss(outputs, batch_y)
                    future_mask = torch.zeros_like(batch_y[:, -args.pred_len:, :])
                    future_mask[:, -max(1, args.pred_len // 4):, :] = 1.0
                    future_align_loss = masked_future_alignment_loss(batch_x, batch_y[:, -args.pred_len:, :], future_mask)
                    loss = pred_loss + args.phase_weight * phase_loss + args.align_weight * future_align_loss
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                pred_loss = criterion(outputs, batch_y)
                if args.loss.upper() == 'MIXED':
                    pred_loss = args.mse_weight * loss_mse(outputs, batch_y) + args.mae_weight * loss_mae(outputs, batch_y)
                phase_loss = phase_alignment_loss(outputs, batch_y)
                future_mask = torch.zeros_like(batch_y[:, -args.pred_len:, :])
                future_mask[:, -max(1, args.pred_len // 4):, :] = 1.0
                future_align_loss = masked_future_alignment_loss(batch_x, batch_y[:, -args.pred_len:, :], future_mask)
                loss = pred_loss + args.phase_weight * phase_loss + args.align_weight * future_align_loss
                train_loss.append(loss.item())
                if epoch == 0 and i == 0:
                    accelerator.print(f"[aux-w] (lp,la)=({args.phase_weight},{args.align_weight}) "
                                      f"pred={pred_loss.item():.5f} phase={phase_loss.item():.5f} "
                                      f"align={future_align_loss.item():.5f}")

            if accelerator.device.type == 'cuda':
                torch.cuda.synchronize()
            forward_time += time.time() - forward_start

            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                train_progress.set_postfix(
                    loss=f"{loss.item():.5f}",
                    speed=f"{speed:.3f}s/it",
                    refresh=False,
                )
                iter_count = 0
                time_now = time.time()

            backward_start = time.time()
            if args.use_amp:
                scaler.scale(loss).backward()
                if accelerator.device.type == 'cuda':
                    torch.cuda.synchronize()
                backward_time += time.time() - backward_start
                optim_start = time.time()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                if accelerator.device.type == 'cuda':
                    torch.cuda.synchronize()
                backward_time += time.time() - backward_start
                optim_start = time.time()
                model_optim.step()
            if accelerator.device.type == 'cuda':
                torch.cuda.synchronize()
            optim_time += time.time() - optim_start
            last_iter_end = time.time()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        epoch_cost = time.time() - epoch_time
        accelerator.print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, epoch_cost))
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        if args.use_seasonal_residual and args.seasonal_alpha >= 0:
            test_loss, test_mae_loss = vali_fold(args, accelerator, model, test_loader, args.seasonal_alpha)
            cname = cand_label(args)
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali MSE: {2:.7f} Vali MAE: {3:.7f} "
                "Test MSE[{6} a={7:.2f}]: {4:.7f} Test MAE[{6} a={7:.2f}]: {5:.7f}".format(
                    epoch + 1, train_loss, vali_loss, vali_mae_loss, test_loss, test_mae_loss, cname, args.seasonal_alpha))
        else:
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali MSE: {2:.7f} Vali MAE: {3:.7f} Test MSE: {4:.7f} Test MAE: {5:.7f}".format(
                    epoch + 1, train_loss, vali_loss, vali_mae_loss, test_loss, test_mae_loss))

        early_stopping(vali_mae_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    best_ckpt = os.path.join(path, 'checkpoint')
    load_best = (args.early_stop == 1)
    if load_best and not os.path.exists(best_ckpt):
        accelerator.print(f'Best checkpoint not found at {best_ckpt}; falling back to last-epoch model')
        load_best = False

    if load_best:
        accelerator.print(f'Loading best checkpoint from {best_ckpt}')
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = torch.load(best_ckpt, map_location=accelerator.device)
        unwrapped_model.load_state_dict(state_dict)
    else:
        accelerator.print('--early_stop 0: evaluating the LAST-epoch model (not best checkpoint)')

    fold_test = args.use_seasonal_residual and args.seasonal_alpha >= 0
    if fold_test:
        test_loss = test_mae_loss = None
    else:
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
    preds, trues, seasonals = [], [], []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), disable=not args.show_progress, **SHORT_BAR_KWARGS):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            if args.use_seasonal_residual:
                cand = posthoc_cand(batch_x, args, f_dim)
                if args.seasonal_alpha >= 0:
                    outputs = (1.0 - args.seasonal_alpha) * outputs + args.seasonal_alpha * cand
                    outs_g, tru_g = accelerator.gather_for_metrics((outputs, batch_y))
                else:
                    outs_g = accelerator.gather_for_metrics((outputs, cand, batch_y))
                    outs_g, cand_g, tru_g = outs_g
                    seasonals.append(cand_g.detach().float().cpu().numpy())
            else:
                outs_g, tru_g = accelerator.gather_for_metrics((outputs, batch_y))
            preds.append(outs_g.detach().float().cpu().numpy())
            trues.append(tru_g.detach().float().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    if fold_test:
        accelerator.print(f'MAE[{cand_label(args)} a={args.seasonal_alpha:.2f}]: {mae:.7f} '
                          f'MSE[{cand_label(args)} a={args.seasonal_alpha:.2f}]: {mse:.7f} RMSE: {rmse:.7f}')
    else:
        accelerator.print(f'Test Loss: {test_loss:.7f} Test MAE Loss: {test_mae_loss:.7f}')
        accelerator.print(f'MAE: {mae:.7f} MSE: {mse:.7f} RMSE: {rmse:.7f} MAPE: {mape:.7f} MSPE: {mspe:.7f}')
    if args.use_seasonal_residual and args.seasonal_alpha >= 0:
        accelerator.print(f'(final result folded with naive 24h residual, alpha={args.seasonal_alpha})')
    if args.use_seasonal_residual and args.seasonal_alpha < 0 and len(seasonals) > 0:
        seasonals = np.concatenate(seasonals, axis=0)
        posthoc_report(preds, seasonals, trues, args.seasonal_alphas, name=cand_label(args))

    accelerator.print('checkpoint preserved')
import argparse
import glob
import json
import os
import statistics
import sys


def load_runs(root):
    runs = []
    for meta_path in glob.glob(os.path.join(root, '**', 'run_meta.json'), recursive=True):
        run_dir = os.path.dirname(meta_path)
        metrics_path = os.path.join(run_dir, 'metrics.json')
        if not os.path.exists(metrics_path):
            print(f'[skip] {run_dir}: no metrics.json (run did not finish)', file=sys.stderr)
            continue
        with open(meta_path, encoding='utf-8') as fh:
            meta = json.load(fh)
        with open(metrics_path, encoding='utf-8') as fh:
            metrics = json.load(fh)
        args = meta.get('args', {})
        runs.append({
            'data': args.get('data'),
            'features': args.get('features'),
            'pred_len': args.get('pred_len'),
            'seed': meta.get('seed', metrics.get('seed')),
            'mse': metrics['test']['mse'],
            'mae': metrics['test']['mae'],
            'best_epoch': metrics.get('best_epoch'),
            'n_params': metrics.get('n_trainable_params'),
        })
    return runs


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser(description='Aggregate PhaseAlign runs into result tables')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints')
    parser.add_argument('--out', type=str, default='', help='optional CSV output path')
    args = parser.parse_args()

    runs = load_runs(args.checkpoints)
    if not runs:
        print(f'no finished runs found under {args.checkpoints}', file=sys.stderr)
        sys.exit(1)

    groups = {}
    for run in runs:
        groups.setdefault((run['data'], run['features'], run['pred_len']), []).append(run)

    rows = []
    for (data, features, pred_len), items in sorted(
            groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2] or 0)):
        seeds = [r['seed'] for r in items]
        mse_mean, mse_std = mean_std([r['mse'] for r in items])
        mae_mean, mae_std = mean_std([r['mae'] for r in items])
        rows.append({
            'dataset': data, 'features': features, 'pred_len': pred_len,
            'n_seeds': len(set(seeds)), 'duplicates': len(seeds) - len(set(seeds)),
            'mse_mean': mse_mean, 'mse_std': mse_std,
            'mae_mean': mae_mean, 'mae_std': mae_std,
            'params': items[0]['n_params'],
            'best_epochs': sorted({r['best_epoch'] for r in items if r['best_epoch'] is not None}),
        })

    def fmt(mean, std):
        return '' if mean is None else f'{mean:.4f} +- {std:.4f}'

    header = ['Dataset', 'H', 'n', 'MSE (mean +- std)', 'MAE (mean +- std)', 'params']
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')
    for r in rows:
        cells = [str(r['dataset']), str(r['pred_len']), str(r['n_seeds']),
                 fmt(r['mse_mean'], r['mse_std']), fmt(r['mae_mean'], r['mae_std']),
                 str(r['params'])]
        print('| ' + ' | '.join(cells) + ' |')

    incomplete = [r for r in rows if r['n_seeds'] < 3]
    if incomplete:
        print('\nIncomplete groups (< 3 seeds):', file=sys.stderr)
        for r in incomplete:
            print(f"  {r['dataset']} H={r['pred_len']}: {r['n_seeds']} seed(s)", file=sys.stderr)

    if args.out:
        with open(args.out, 'w', encoding='utf-8', newline='') as fh:
            fh.write('dataset,features,pred_len,n_seeds,mse_mean,mse_std,'
                     'mae_mean,mae_std,params,best_epochs\n')
            for r in rows:
                fh.write(','.join(str(x) for x in [
                    r['dataset'], r['features'], r['pred_len'], r['n_seeds'],
                    r['mse_mean'], r['mse_std'], r['mae_mean'], r['mae_std'], r['params'],
                    ' '.join(str(e) for e in r['best_epochs']),
                ]) + '\n')
        print(f'\nwrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()

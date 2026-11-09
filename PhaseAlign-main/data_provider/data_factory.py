import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'exchange': Dataset_Custom,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'illness': Dataset_Custom,
}


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    torch.manual_seed(seed % (2 ** 32))


def data_provider(args, flag, seed=0):
    if args.data not in data_dict:
        raise KeyError(f'unknown --data {args.data!r}; available: {sorted(data_dict)}')
    Data = data_dict[args.data]

    shuffle_flag = flag == 'train'
    drop_last = flag == 'train'
    batch_size = args.batch_size if flag == 'train' else args.eval_batch_size

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        freq=args.freq,
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed) + (0 if flag == 'train' else 1))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=_worker_init_fn if args.num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    return data_set, data_loader

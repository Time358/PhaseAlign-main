import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

VALID_FEATURES = ('M', 'S', 'MS')


class _BaseDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S',
                 data_path='ETTh1.csv', target='OT', freq='h'):
        if size is None:
            size = [24 * 4 * 4, 24 * 4, 24 * 4]
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ('train', 'val', 'test'), f'unknown split {flag!r}'
        assert features in VALID_FEATURES, f'features must be one of {VALID_FEATURES}'
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features, self.target, self.freq = features, target, freq
        self.root_path, self.data_path = root_path, data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def _borders(self, n_rows):
        raise NotImplementedError

    def _reorder_target_last(self, df_raw):
        return df_raw

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = self._reorder_target_last(df_raw)

        border1s, border2s = self._borders(len(df_raw))
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        if self.features in ('M', 'MS'):
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
        data = self.scaler.transform(df_data.values)

        df_stamp = df_raw[['date']][border1:border2].copy()
        dates = pd.DatetimeIndex(pd.to_datetime(df_stamp['date'].values))
        data_stamp = time_features(dates, freq=self.freq).transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.tot_len * self.enc_in


class _ETTDataset(_BaseDataset):
    steps_per_hour = 1

    def _borders(self, n_rows):
        h = 30 * 24 * self.steps_per_hour
        border1s = [0, 12 * h - self.seq_len, 16 * h - self.seq_len]
        border2s = [12 * h, 16 * h, 20 * h]
        if n_rows < border2s[-1]:
            raise ValueError(f'{self.data_path} has {n_rows} rows, the ETT 12/4/4 protocol '
                             f'needs {border2s[-1]}')
        return border1s, border2s


class Dataset_ETT_hour(_ETTDataset):
    steps_per_hour = 1


class Dataset_ETT_minute(_ETTDataset):
    steps_per_hour = 4


class Dataset_Custom(_BaseDataset):
    def _borders(self, n_rows):
        num_train = int(n_rows * 0.7)
        num_test = int(n_rows * 0.2)
        num_vali = n_rows - num_train - num_test
        border1s = [0, num_train - self.seq_len, n_rows - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, n_rows]
        return border1s, border2s

    def _reorder_target_last(self, df_raw):
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        return df_raw[['date'] + cols + [self.target]]

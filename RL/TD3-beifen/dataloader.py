import torch
import numpy as np

class DataLoader:

    def __init__(self, df_feature, df_label, batch_size=800, pin_memory=False, start_index = 0, device=None):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature #.values
        self.df_label = df_label #.values
        self.device = device
        # self.time_index = time_index

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)
            
        self.index = df_label.index

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.start_index = start_index

        self.daily_count = df_label.groupby(level=0).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    @property
    def batch_length(self):

        if self.batch_size <= 0:
            return self.daily_length

        return len(self.df_label) // self.batch_size

    @property
    def daily_length(self):

        return len(self.daily_count)

    def iter_batch(self):
        if self.batch_size <= 0:
            yield from self.iter_daily_shuffle()
            return

        indices = np.arange(len(self.df_label))
        np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield i, indices[i:i+self.batch_size] # NOTE: advanced indexing will cause copy

    def iter_daily_shuffle(self):
        indices = np.arange(len(self.daily_count))
        np.random.shuffle(indices)
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])

    def iter_daily(self):
        indices = np.arange(len(self.daily_count))
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        # for idx, count in zip(self.daily_index, self.daily_count):
        #     yield slice(idx, idx + count) # NOTE: slice index will not cause copy

    def get(self, dt, st):
        # day = self.time_index[slc]
        # df_label_today = self.df_label.loc[dt,st]
        # stock_today = df_label_today.index
        # outs = self.df_feature[slc], self.df_label[slc] #[:,0],
        # outs = self.df_feature[slc], self.df_label[slc]
        # print(dt,st)
        outs = self.df_feature.loc[dt,st].values, self.df_label.loc[dt,st] #[:,0],
        # if not self.pin_memory:
        outs = tuple(torch.tensor(x, dtype=torch.float,).unsqueeze(0) for x in outs)  #  device=self.device

        return outs #+ (self.index[dt], st, dt,)

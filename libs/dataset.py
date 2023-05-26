# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class DataSets:

    def __init__(self, raw_df):
        self.raw = raw_df
        self.tdf = None

        self._y_scale = None
        self._y_mean = None

    def seektime(self, time, timedelta):
        start = pd.Timestamp(time)
        self.tdf = self.raw.loc[start: start+timedelta]

    def scale(self):
        scaler = StandardScaler()
        train_df, _, _, _ = self.split_data(self.tdf)
        ret = scaler.fit(train_df)

        self._y_mean, self._y_scale = scaler.mean_[0], scaler.scale_[0]
        print(scaler.scale_[0], scaler.mean_[0])

        self.tdf = pd.DataFrame(scaler.transform(self.tdf), index=self.tdf.index, columns=self.tdf.columns)
        self.tdf = self.tdf.fillna(0)

        return self.tdf

    def preprocess(self, time, timedelta):
        self.raw.index = pd.to_datetime(self.raw.index)
        self.seektime(time, timedelta)
        self.scale()

        return self.tdf

    def split_data(self, df, train_rate=0.75, valid_rate=0.05):
        #print("===== 5. split data =====")
        trn_len = len(df)
        vld_len = int(len(df) * valid_rate)

        train_df = df.iloc[:trn_len,:]
        valid_df = df.iloc[-(vld_len*2):-vld_len,:]
        test_df = df.iloc[-vld_len:,:]

        np_tr_x = train_df
        np_tr_y = train_df.iloc[0, :]
        
        return np_tr_x, np_tr_y, valid_df, test_df


class WindowGenerator():
    def __init__(self, input_width:int, label_width:int,
                 train_df:pd.DataFrame, valid_df:pd.DataFrame, test_df:pd.DataFrame,
                 shift=0, label_col_index=0, verbose=1):
        self.label_col_index = label_col_index
        
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        
        self.input_width = input_width # window_size
        self.label_width = label_width # step_size
        self.shift = shift
        
        self.total_input_width = input_width + shift
        self.total_width = input_width + shift + label_width

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_input_width)[self.input_slice]

        self.label_start = self.total_input_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_width)[self.label_slice]

        self.train = self.make_dataset(self.train_df, shuffle_=True)
        self.val = self.make_dataset(self.valid_df, shuffle_=False)
        self.test = self.make_dataset(self.test_df, shuffle_=False)

        self.train_times = self.train_df.iloc[self.total_input_width:-(self.label_width-1)].index # current_time point
        self.val_times = self.valid_df.iloc[self.total_input_width:-(self.label_width-1)].index # current_time point
        self.test_times = self.test_df.iloc[self.total_input_width:-(self.label_width-1)].index # current_time point

        self.verbose = verbose

    def split_window(self, sequence):
        inputs = sequence[:,self.input_slice,:]  # inputs.shape: (batch, window_size, num_features)
        # label_col은 항상 0으로 고정시켰음!!
        labels = sequence[:,self.label_slice,0:1] # labels.shape: (batch, step_size, 1)
        # labels = sequence[:,self.label_slice,self.label_col_index:(self.label_col_index+1)] # labels.shape: (batch, step_size, 1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, df, shuffle_=False):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=df,
            targets=None,
            sequence_length=self.total_width,
            shuffle=shuffle_,
            batch_size=16,
        )
        ds = ds.map(self.split_window) # (None, [(batch, window_size, num_features), (batch, step_size)])
        return ds

    def sample(self, df=None, n=4, numpy_seed=None):
        if numpy_seed is not None:
            np.random.seed(numpy_seed)
        if df is None:
            df = self.test_df
            times = self.test_times
        else:
            times = df.iloc[self.total_input_width:-(self.label_width-1)].index # current_time point

        print(times)
        times_idxs = list(np.random.choice(len(times), min(n,len(times)), replace=False))
        
        example_window = tf.stack([np.array(df[t:t+self.total_width]) for t in times_idxs])
        example_times = df.index.astype(str).values[times_idxs]
        example_inputs, example_labels = self.split_window(example_window)
        
        if self.verbose>0:
            print('All shapes are: (batch, time, features)')
            # print(f'* Window shape: {example_window.shape}')
            # print(f'* [Inputs shape]: {example_inputs.shape}') # ex. <class 'tensorflow.python.framework.ops.EagerTensor'> (32, 96, 2)
            # print(f'* [labels shape]: {example_labels.shape}') # ex. <class 'tensorflow.python.framework.ops.EagerTensor'> (32, 96, 1)
            # print(f'* example_times: {example_times}')

        return example_inputs, example_labels, example_times

    def __repr__(self):
        return '\n'.join([
            f'[1] Total width (per Sequence): {self.total_width}',
            f'* Input indices: {self.input_indices}',
            f'* Shift: {self.shift}',
            f'* Label indices: {self.label_indices}',
            f'[2] Label column Index: {self.label_col_index}'])

def ds_shape(dataset):
    dataset_to_numpy = list(dataset.as_numpy_iterator())
    return tf.shape(dataset_to_numpy)

def load_dataset(dataset_path, days=10, start_date="2021-03-17"):
    df = pd.read_csv(dataset_path, index_col=0)
    ds = DataSets(df)

    df = ds.preprocess(start_date, pd.Timedelta(days=days))
    tr_x, tr_y, vd, tt = ds.split_data(df)

    return tr_x, tr_y, vd, tt

if __name__ == "__main__":
    #ds = load_dataset()
    tr, _, vd, tt = load_dataset(days=10)

    wg = WindowGenerator(input_width=9, label_width=1, train_df=tr, valid_df=vd, test_df=tt, shift=10, label_col_index=0)
    #ds = wg.make_dataset(tr)
    #print(ds)



#curl https://github.com/unit8co/sds2023-forecating-and-meta-learning/blob/main/data/m3_dataset.xls\?raw\=true -o m3_dataset.xls
#curl https://github.com/unit8co/sds2023-forecating-and-meta-learning/blob/main/data/passengers_per_carrier.csv\?raw\=true -o passengers_per_carrier.csv
#curl https://github.com/unit8co/sds2023-forecating-and-meta-learning/blob/main/data/m4_monthly_scaled.pkl\?raw\=true -o m4_monthly_scaled.pkl

import warnings
warnings.filterwarnings('ignore')

import os
import time
import random
import pandas as pd
import pickle
import numpy as np
import requests
import zipfile
import tqdm.notebook as tq
from datetime import datetime
import torch
from torch import nn
from typing import List, Tuple, Dict
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.callbacks import Callback, EarlyStopping

from darts import TimeSeries
from darts.utils.losses import SmapeLoss
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import smape, mase, mape
from darts.utils.data import HorizonBasedDataset
from darts.utils.utils import SeasonalityMode, TrendMode, ModelMode
from darts.models import (
    NaiveSeasonal, NBEATSModel, ExponentialSmoothing,
    TCNModel, RegressionModel, LinearRegressionModel,
    LightGBMModel, ARIMA, Theta, KalmanForecaster, NHiTSModel
)

HORIZON = 18

def load_air() -> Tuple[List[TimeSeries], List[TimeSeries]]:
    df_per_carrier = pd.read_csv(r'..\\data\\meta-learning\\passengers_per_carrier.csv', dtype={"passengers": np.float32})
    df_per_carrier["month"] = df_per_carrier["month"].apply(pd.to_datetime)
    # Passenger filtering
    carriers_total_passenger = df_per_carrier.groupby("carrier_name").agg(total_passengers = ("passengers", "sum"))
    CARRIERS_TO_KEEP = carriers_total_passenger.query("total_passengers > 100000").index.values.tolist()
    df_per_carrier_filt = (
        df_per_carrier
        .query("carrier_name in @CARRIERS_TO_KEEP")
    )
    df_per_carrier_filt = df_per_carrier_filt.loc[:, ["carrier_name", "passengers", "month"]]
    min_len = 60
    air_data = TimeSeries.from_group_dataframe(df_per_carrier_filt, group_cols="carrier_name", value_cols="passengers", time_col="month", freq="MS")
    # Filtering
    air_data = [a for a in air_data if len(a) > min_len and a.min(axis=0).values() > 0]
    # Interpolating
    transformer = MissingValuesFiller()
    air_data = [transformer.transform(a) for a in air_data]
    # Train test
    air_train, air_test = [a[:-HORIZON] for a in air_data], [a[-HORIZON:] for a in air_data]
    # Rescaling between 0 and 1
    scaler_air = Scaler(scaler=MaxAbsScaler())
    air_train = scaler_air.fit_transform(air_train)
    air_test = scaler_air.transform(air_test)
    print('done. There are {} series, with average training length {}'.format(
          len(air_train), np.mean([len(s) for s in air_train])
      ))
    return air_train, air_test

def eval_forecasts(pred_series: List[TimeSeries],
                   test_series: List[TimeSeries]) -> List[float]:

    print('computing sMAPEs...')
    smapes = smape(test_series, pred_series)
    mean, std = np.mean(smapes), np.std(smapes)
    print('Avg sMAPE: %.3f +- %.3f' % (mean, std))
    plt.figure(figsize=(4,4), dpi=144)
    plt.hist(smapes, bins=50)
    plt.ylabel('Count')
    plt.xlabel('sMAPE')
    plt.show()
    plt.close()
    return smapes

def eval_local_model(train_series: List[TimeSeries],
                     test_series: List[TimeSeries],
                     model_cls,
                     **kwargs) -> Tuple[List[float], float]:
    preds = []
    start_time = time.time()
    for series in train_series:
        model = model_cls(**kwargs)
        model.fit(series)
        pred = model.predict(n=HORIZON)
        preds.append(pred)
    elapsed_time = time.time() - start_time
    smapes = eval_forecasts(preds, test_series)
    return smapes, elapsed_time
def eval_global_model(train_series: List[TimeSeries],
                      test_series: List[TimeSeries],
                      model_cls,
                      **kwargs) -> Tuple[List[float], float]:

    start_time = time.time()

    # build your model here
    ...

    # fit your model here
    ...

    # get some predictions here
    preds = ...

    elapsed_time = time.time() - start_time

    smapes = eval_forecasts(preds, test_series)
    return smapes, elapsed_time

#data = TimeSeries.from_csv('..\data\meta-learning\passengers_per_carrier.csv')

air_train, air_test = load_air()
print(air_train[0])

# for i in [1, 20, 50, 100, 125]:
#     plt.figure(figsize=(4,4), dpi=144)
#     air_train[i].plot(label=air_train[i].static_covariates.loc["passengers", "carrier_name"])
#     plt.ylabel('Passengers')
#     plt.xlabel('Time')
#     plt.show()
#     plt.close()
#
#
# # Naive model
# naive_seasonal_last_smapes, naive_seasonal_last_elapsed_time = eval_local_model(air_train, air_test, NaiveSeasonal, K=1)
# lessnaive_seasonal_last_smapes, lessnaive_seasonal_last_elapsed_time = eval_local_model(air_train, air_test, NaiveSeasonal, K=12)


# ------------- Deep Learning -----------
# Slicing hyper-params:
IN_LEN = 24
OUT_LEN = 12

# Architecture hyper-params:
NUM_STACKS = 18
NUM_BLOCKS = 3
NUM_LAYERS = 3
LAYER_WIDTH = 180
COEFFS_DIM = 6
LOSS_FN = SmapeLoss()

# Training settings:
LR = 5e-4
BATCH_SIZE = 1024
NUM_EPOCHS = 4

# reproducibility
np.random.seed(42)
torch.manual_seed(42)

## Use this to specify "optimizer_kwargs" parameter of the N-BEATS model:
optimizer_kwargs={'lr': LR},

## In addition, when using a GPU, you should specify this for
## the "pl_trainer_kwargs" parameter of the N-BEATS model:
# pl_trainer_kwargs={"enable_progress_bar": True,
#                    "accelerator": "gpu",
#                    "gpus": -1,
#                    "auto_select_gpus": True}
#
# start_time = time.time()
#
# nbeats_model_air = NBEATSModel() # Build the N-BEATS model here
#
# nbeats_model_air.fit(..., # fill in series to train on
#                      ...) # fill in number of epochs
#
# # get predictions
# nb_preds = ...
#
# nbeats_smapes = eval_forecasts(nb_preds, air_test)
# nbeats_elapsed_time = time.time() - start_time



# --------------- META-LEARNING ---------------------
def load_m4() -> Tuple[List[TimeSeries], List[TimeSeries]]:
    # load TimeSeries - the splitting and scaling has already been done
    print(r'..\\data\\meta-learning\\loading M4 TimeSeries...')
    with open(r'..\\data\\meta-learning\\m4_monthly_scaled.pkl', 'rb') as f:
        m4_series = pickle.load(f)
    m4_train_scaled, m4_test_scaled = zip(*m4_series)

    print('done. There are {} series, with average training length {}'.format(
        len(m4_train_scaled), np.mean([len(s) for s in m4_train_scaled])
    ))
    return m4_train_scaled, m4_test_scaled

#m4_train, m4_test = load_m4()
# filter to keep only those that are long enough
#filtered = filter(lambda t: len(t[0]) >= 48, zip(m4_train, m4_test))
#m4_train, m4_test = zip(*filtered)
#m4_train, m4_test = list(m4_train), list(m4_test)

#print('There are {} series of length >= 48.'.format(len(m4_train)))


# Slicing hyper-params:
IN_LEN = 36
OUT_LEN = 12

# Architecture hyper-params:
NUM_STACKS = 18
NUM_BLOCKS = 3
NUM_LAYERS = 3
LAYER_WIDTH = 180
COEFFS_DIM = 6
LOSS_FN = SmapeLoss()

# Training settings:
LR = 5e-4
BATCH_SIZE = 1024
MAX_SAMPLES_PER_TS = 8   # <-- new param, limiting nr of training samples per epoch
NUM_EPOCHS = 7

# Pretrained NBEATS model
with zipfile.ZipFile(r"..\models\nbeats_pretrained_model_m4\nbeats\NBEATSModel.pt","r") as zip_ref:
    zip_ref.extractall("/content/")
nbeats_model_m4 = NBEATSModel.load(r"..\models\nbeats_pretrained_model_m4\nbeats\NBEATSModel.pt")
nbeats_model_m4.to_cpu()

start_time = time.time()
preds = nbeats_model_m4.predict(n=HORIZON, series=air_train) # get forecasts
nbeats_m4_smapes = eval_forecasts(preds, air_test)

# one shot learning on the air dataset
nbeats_model_m4.fit(air_train, )

nbeats_m4_elapsed_time = time.time() - start_time



# ----------------- NHITS Model -------------------

nhits = NHiTSModel(input_chunk_length=IN_LEN, output_chunk_length=OUT_LEN,
                   num_stacks=NUM_STACKS, num_blocks=NUM_BLOCKS,
                   num_layers=NUM_LAYERS, layer_widths=LAYER_WIDTH)

nhits.fit(air_train)
preds = nhits.predict(n=HORIZON, series=air_train)
nhits_m4_smapes = eval_forecasts(preds, air_train)

print("finished")
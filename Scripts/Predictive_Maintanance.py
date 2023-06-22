# Auto encoder: problem of residual spill over of reconstruction models -> can get residual in the inputs where no residual is expected but non in the target
# >> thats why rather use regression model: inputs without expected failures are mapped to targets where failure is possible
# reproduces not only statistics both als physical workings of the system
# Failure score is probability from the distribution of residuals of the model


import os
import h5py
import time
import random
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import DataFrame
from itertools import product
from matplotlib import gridspec
from operator import itemgetter
from time import gmtime, strftime
from sklearn.model_selection import train_test_split


def plot_variables_iso(data, figsize=10, filename=None,
                       labelsize=16, y_min=None, y_max=None, x_min=None, x_max=None):
    """
    Given a "data" dictionary it generates a plot of size 'figsize' and 'labelsize'.
    If the filename is provided, the resulting plot is saved.

    Expected keys: 'variables', 'ds_name', 'legend.'
        data[0]['variables']: contains a list with the variables' names that should be plotted
        data[0]['ds_name']: contains the data subsets to be plotted
        data[0]['legend']: contains LaTex formatted legend of each plot

    Optional keys:
        data[0]['xlabel'] = e.g. 'Time [cycles]' , default: 'Index'
        data[0]['ylabel']: If data[0]['ylabel']='score' then ylabel is 'Anomaly score' and title
        data[0][jj]['marker'] (default marker '.')
        data[0][jj]['units']

    Plotting variables are provided as:
        data[0][jj]['x']
        data[0][jj]['y']

        """
    plt.clf()

    input_dim = len(data[0]['variables'])
    cols = min(np.floor(input_dim ** 0.5).astype(int), 4)
    rows = (np.ceil(input_dim / cols)).astype(int)
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(figsize, max(figsize, rows * 2)))

    color_dic = {'dev': 'C0', 'train': 'C0', 'lab': 'C0', 'val': 'C1', 'unl': 'C2', 'test': 'C3',
                 '1': 'C0', '2': 'C1', '3': 'C2', '4': 'C3', '5': 'C4', '6': 'C5', '7': 'C6', '8': 'C7',
                 '9': 'C8', '10': 'C9', '11': 'C1', '12': 'C11', '13': 'C12', '14': 'C2', '15': 'C3',
                 '16': 'C15', '17': 'C16', '18': 'C17', '19': 'C18', '20': 'C19'}

    # Plot dataset types
    for n in range(input_dim):
        ax = fig.add_subplot(gs[n])
        for jj in data[0]['ds_name']:
            if 'units' in data[0][jj]:
                for unit in np.unique(data[0][jj]['units']):
                    mask = np.ravel(data[0][jj]['units'] == unit)
                    ax.plot(data[0][jj]['x'][mask], data[0][jj]['y'][mask, n], '.',
                            color=color_dic[str(int(unit))],
                            markeredgewidth=0.25, markersize=8)
            else:
                if 'marker' in data[0][jj]:
                    ax.plot(data[0][jj]['x'], data[0][jj]['y'][:, n], data[0][jj]['marker'],
                            markeredgewidth=0.25, markersize=8)
                else:
                    ax.plot(data[0][jj]['x'], data[0][jj]['y'][:, n], '.',
                            markeredgewidth=0.25, markersize=8)

                    # Axis adjusments (max, min values, labelsize and rotations)
        if y_min != None:
            ax.set_ylim(bottom=y_min)
        if y_max != None:
            ax.set_ylim(top=y_max)
        if x_max != None:
            ax.set_xlim(0, x_max)
        ax.tick_params(axis='x', labelsize=labelsize)  # rotation=45
        ax.tick_params(axis='y', labelsize=labelsize)

        # Labels
        if 'xlabel' in data[0]:
            plt.xlabel(data[0]['xlabel'], fontsize=labelsize)
        else:
            plt.xlabel('Index', fontsize=labelsize)
        if 'ylabel' in data[0]:
            if data[0]['ylabel'] == 'score':
                plt.title(data[0]['variables'][n], fontsize=labelsize)
                plt.ylabel('Anomaly Score', fontsize=labelsize)
            else:
                plt.ylabel(data[0]['ylabel'][n], fontsize=labelsize)
        else:
            plt.ylabel(data[0]['variables'][n], fontsize=labelsize)

        # Legend
        leg = []
        for jj in data[0]['ds_name']:
            if 'units' in data[0][jj]:
                for u in np.unique(data[0][jj]['units']):
                    leg.append('Unit ' + str(int(u)))
            elif (('units' not in data[0][jj]) and ('legend' in data[0])):
                leg = data[0]['legend']
        plt.legend(leg, fontsize=labelsize - 2, loc='best')

    # draw solid white grid lines
    plt.grid(color='w', linestyle='solid')

    plt.tight_layout()

    if filename == None:
        plt.show()
    else:
        print(filename + '.png')
        plt.savefig(filename + '.png', format='png', dpi=300)
    plt.close()


color_dic_unit = {'Unit 1': 'C0', 'Unit 2': 'C1', 'Unit 3': 'C2', 'Unit 4': 'C3', 'Unit 5': 'C4', 'Unit 6': 'C5',
                  'Unit 7': 'C6', 'Unit 8': 'C7', 'Unit 9': 'C8', 'Unit 10': 'C9', 'Unit 11': 'C1',
                  'Unit 12': 'C11', 'Unit 13': 'C12', 'Unit 14': 'C2', 'Unit 15': 'C3', 'Unit 16': 'C15',
                  'Unit 17': 'C16', 'Unit 18': 'C17', 'Unit 19': 'C18', 'Unit 20': 'C19',
                  'dev': 'C0', 'threshold': 'k'}


def subplot_per_unit(data, color_dic):
    """
    Creates subplots of time-series data for each unit, with vertical lines indicating known faults.

    Parameters:
        data (list): A list of dictionaries containing time-series data for each unit.
                     Each dictionary should have the following keys:
                        - 'ds_name': A string name for the data segment
                        - 'units': A list of integers indicating the unit numbers associated with this data segment
                        - 'x': A list or numpy array of timestamps for the data
                        - 'y': A list or numpy array of values for the data
                        - 'fault' (optional): A timestamp indicating a known fault in the data segment
                        - 'xlabel': A string label for the x-axis
                        - 'ylabel': A string label for the y-axis
        color_dic (dict): A dictionary mapping unit numbers to colors, to be used in plotting the data.
                          The keys should be strings of unit numbers (e.g. '1', '2', '3') and the values
                          should be valid matplotlib color strings (e.g. 'red', 'blue', 'green').

    Returns:
        None

    """
    plt.clf()
    rows, cols = [len(data), 1]
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(10, 10))
    for n in range(rows):
        ax = fig.add_subplot(gs[n])
        leg = []
        for j in data[ii]['ds_name']:
            # Plot data segments
            unit = data[n][j]['units'][0]
            ax.plot(data[n][j]['x'], data[n][j]['y'], linestyle='None', marker='.', color=color_dic[str(unit)])

            # Plot vertical lines
            if 'fault' in data[n][j]:
                plt.plot([data[n][j]['fault'], data[n][j]['fault']], [0, 1], 'k',
                         markerfacecolor='none', linewidth=2, linestyle='--')

        # legend
        leg.append('Unit ' + str(unit))

        # Adjustments
        plt.yticks([0, 1])
        plt.ylabel(data[n]['ylabel'], fontsize=16)
        plt.xlabel(data[n]['xlabel'], fontsize=16)
        plt.legend(leg, loc='best', fontsize=15)
        # plt.title(data[n]['title'])
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    fig.tight_layout()

# -------------------------- Preprocessing --------------------------------

def extract_units_ds(id_en, ds, units):
    '''
    Creates a subset with only id_en units for ds

    Parameters:
        id_en (list): A list of unit ids to be extracted.
        ds (numpy array): A 2D numpy array of data with shape (n_samples, n_features)
        units (numpy array): A 1D numpy array of length n_samples containing the unit id for each sample.

    Returns:
        numpy array: A 2D numpy array of shape (n_samples_sub, n_features) containing only the samples corresponding to the units in id_en.

    '''

    # Set-up
    ds_sub = []
    units_unique = np.unique(units)

    # Process
    for i in units_unique:
        if i in id_en:
            idx = np.ravel(units == i)
            ds_sub.append(ds[idx, :])

    return np.concatenate(ds_sub, axis=0)


def normalize_data(x, lb, ub, max_v=1.0, min_v=-1.0):
    """
    Normalize data using min-max normalization

    Parameters:
        x (numpy array): A 2D numpy array of data with shape (n_samples, n_features).
        lb (numpy array): A 1D numpy array of length n_features containing the lower bounds of each feature. If an empty list is passed, the function will compute the lower bounds based on the minimum values of each feature.
        ub (numpy array): A 1D numpy array of length n_features containing the upper bounds of each feature. If an empty list is passed, the function will compute the upper bounds based on the maximum values of each feature.
        max_v (float): The maximum value to be used for normalization. Default is 1.0.
        min_v (float): The minimum value to be used for normalization. Default is -1.0.

    Returns:
        tuple: A tuple containing the normalized data as a 2D numpy array with shape (n_samples, n_features), the computed lower bounds as a 1D numpy array with shape (1, n_features), and the computed upper bounds as a 1D numpy array with shape (1, n_features).

    """

    # Set-up
    if len(ub) == 0:
        ub = x.max(0)  # OPTION 1
        # ub = np.percentile(x, 99.9, axis=0, keepdims=True) # OPTION 2:

    if len(lb) == 0:
        lb = x.min(0)
        # lb = np.percentile(x, 0.1, axis=0, keepdims=True)

    ub.shape = (1, -1)
    lb.shape = (1, -1)
    max_min = max_v - min_v
    delta = ub - lb

    # Compute
    x_n = max_min * (x - lb) / delta + min_v
    if 0 in delta:
        idx = np.ravel(delta == 0)
        x_n[:, idx] = x[:, idx] - lb[:, idx]

    return x_n, lb, ub


# Split available dataset
def data_subset(X_data, Units, Cycles, U_sel, split_cycle=10):
    """
    Creates two dataset subsets. One containing labeled (healthy) data and another
    with unlabeled data

    Arguments:
    X_data -- np.array() with Xs data of shape (number of examples, number of features)
    U_sel -- list with selected unit number labels
    Units -- np.array() with unit number labels
    Cycles -- np.array() with flight cycles numbers

    Returns:
    X_lab, X_unl -- np.arrays for {lab, unl}
    """

    # Set-up
    X_lab, X_unl = [], []

    # Loop over units
    for u in U_sel:
        unit = np.ravel(Units == u)
        X_unit = X_data[unit, :]
        C_unit = Cycles[unit, :]

        # Labeled healthy dataset
        mask = np.ravel(C_unit <= split_cycle)
        X_lab.append(X_unit[mask, :])

        # Unlabeled dataset
        X_unl.append(X_unit[~mask, :])

    return np.vstack(X_lab), np.vstack(X_unl)


# ------------------- Read Data -------------------------------

ROOT_PATH = r"..\\"  # You need to set your own path here
PATH_IN = ROOT_PATH + r"data\\PredictiveMaintanance"
SOURCE = 'CMAPSS_Dataset_DS02_Journal'

# Time tracking, Operation time (min):  0.004
t = time.perf_counter()

with h5py.File(PATH_IN + "/" + SOURCE + '.h5', 'r') as hdf:
    # Nominal Development set
    W_dev = np.array(hdf.get('W_train'))  # W , Operative conditions
    Xs_dev = np.array(hdf.get('X_s_train'))  # X_s, Sensor readings
    T_dev = np.array(hdf.get('T_train'))  # T, Performance gaps
    R_dev = np.array(hdf.get('Y_train'))  # RUL, Remaining Useful Lifetime
    U_dev = np.array(hdf.get('U_train'))  # Units
    C_dev = np.array(hdf.get('C_train'))  # Cycles

    # Nominal Test set - Past
    W_test = np.array(hdf.get('W_test'))  # W
    Xs_test = np.array(hdf.get('X_s_test'))  # X_s
    T_test = np.array(hdf.get('T_test'))  # T
    R_test = np.array(hdf.get('Y_test'))  # RUL
    U_test = np.array(hdf.get('U_test'))  # Units
    C_test = np.array(hdf.get('C_test'))  # Cycles

# Variable name
W_var = ['alt', 'Mach', 'TRA', 'T2']
Xs_var = ['T24', 'T30', 'T40', 'T48', 'T50',
          'P15', 'P2', 'P21', 'P24', 'Ps30', 'P30', 'P40', 'P50',
          'Nf', 'Nc', 'Wf']
T_var = ['HPT_eff_mod', 'LPT_eff_mod', 'LPT_flow_mod']

# Report dataset shapes
print('')
print("number of development data examples = " + str(Xs_dev.shape[0]))
print("Xs_dev shape: " + str(Xs_dev.shape))
print("T_dev shape: " + str(T_dev.shape))
print("W_dev shape: " + str(W_dev.shape))
print("U_dev shape: " + str(U_dev.shape))
print("C_dev shape: " + str(C_dev.shape))

print('')
print("number of test data examples = " + str(Xs_test.shape[0]))
print("Xs_test shape: " + str(Xs_test.shape))
print("T_test shape: " + str(T_test.shape))
print("W_test shape: " + str(W_test.shape))
print("U_test shape: " + str(U_test.shape))
print("C_test shape: " + str(C_test.shape))

print('')
print("Operation time (min): ", (time.perf_counter() - t) / 60)
print('')


# Development units
t_EOL_dev = []
for i in np.unique(U_dev):
    t_EOL_dev = t_EOL_dev + [int(C_dev[U_dev == i][-1])]
    print('Unit: ' + str(int(i)) + ' - Number of flight cyles (t_{EOL}): ', int(C_dev[U_dev == i][-1]))

# Test units
print('')
t_EOL_test = []
for i in np.unique(U_test):
    t_EOL_test = t_EOL_test + [int(C_test[U_test == i][-1])]
    print('Unit: ' + str(int(i)) + ' - Number of flight cycles (t_{EOL}): ', int(C_test[U_test == i][-1]))


# Sub set of data
dim, size = W_dev.shape[0], 10000
mask_dev = np.sort(np.random.choice(dim, size, replace=False))

df_W_dev = pd.DataFrame(W_dev[mask_dev], columns=W_var)
df_W_dev['source'] = U_dev[mask_dev]
for unit in np.unique(U_dev[mask_dev]):
    mask_u = np.ravel(U_dev[mask_dev] == unit)
    df_W_dev.loc[mask_u, 'source'] = 'Unit ' + str(unit)

dim, size = W_test.shape[0], 10000
mask_test = np.sort(np.random.choice(dim, size, replace=False))

df_W_test = pd.DataFrame(W_test[mask_test], columns=W_var)
df_W_test['source']= U_test[mask_test]
for unit in np.unique(U_test[mask_test]):
    mask_u = np.ravel(U_test[mask_test] == unit)
    df_W_test.loc[mask_u, 'source'] = 'Unit ' + str(unit)

df_W = pd.concat([df_W_dev, df_W_test], ignore_index=True)

# Plot
sns.set(font_scale=1.4)
sns.pairplot(df_W, hue='source', palette=color_dic_unit)
import numpy as np

def get_log_returns(data):
    pass

def series_to_sequence(data, seq_length):
    # transforms data of shape [T, D] (T time steps, D features)
    # to shape [T-seq_length, D, seq_length]
    x_seq = []
    for t in range(seq_length, x.shape[0]):
        x_seq_single = []
        for iFt in range(x.shape[1]):
            x_seq_single.append(x[t - seq_length:t, iFt].tolist())
        x_seq.append(x_seq_single)
    x_seq = np.array(x_seq)
    x_seq = np.transpose(x_seq, axes=(0, 2, 1))
    return x_seq
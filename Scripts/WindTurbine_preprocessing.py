import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime as dt

ENDDATE_FIRSTYEAR = dt.datetime(2017, 9, 30)

def preprocess(df):
    df = df.drop(df.columns[0], axis=1)
    df = df.set_index("datetime", drop=True)
    df.index = pd.to_datetime(df.index)
    df = df.dropna()

    sc = StandardScaler()
    df.iloc[:, :] = sc.fit_transform(df)

    indexIsFirstyear = df.index < ENDDATE_FIRSTYEAR

    X_base = df.loc[indexIsFirstyear].iloc[:, :-1]
    y_base = df.loc[indexIsFirstyear].iloc[:, -1]
    X_test = df.loc[~indexIsFirstyear].iloc[:, :-1]
    y_test = df.loc[~indexIsFirstyear].iloc[:, -1]

    return X_base, y_base, X_test, y_test
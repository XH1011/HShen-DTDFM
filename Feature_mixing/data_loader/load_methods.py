import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pickle


def CWRU(item_path):
    axis = ["_DE_time", "_FE_time", "_BA_time"]
    datanumber = os.path.basename(item_path).split(".")[0]
    if eval(datanumber) < 100:
        realaxis = "X0" + datanumber + axis[0]
    else:
        realaxis = "X" + datanumber + axis[0]
    signal = loadmat(item_path)[realaxis]

    return signal


def PU(item_path):
    name = os.path.basename(item_path).split(".")[0]
    fl = loadmat(item_path)[name]
    signal = fl[0][0][2][0][6][2]  #Take out the data
    signal = signal.reshape(-1,1)

    return signal


def GS(item_path):
    signal = pd.read_pickle(item_path)


    return signal



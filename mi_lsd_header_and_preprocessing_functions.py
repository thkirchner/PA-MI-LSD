import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import glob
import time
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import imageio
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.optimize import least_squares
from scipy.stats import iqr
from torch.autograd import Variable
from pathlib import Path
from matplotlib.ticker import MultipleLocator

def preprocess_np_for_milsd(np_path, start_wl_id=0, stop_wl_id=16, 
                            ill_pos_count=4, sort=True, data_augment=False):
    """ Returns numpy arrays of labels y and their feature vectors X; 
    For multiple illumination learned spectral decoloring (MI-LSD).
    The feature vectors X are ill_pos_count of L1 normalized spectra 
    sorted after their decreasing L1 norms.
    A custom range of wavelengths can be selected by specifying 
    start_wl_id and stop_wl_id accordingly.
    
    Keyword arguments:
    np_path -- string path to the numpy arrays to be preprocessed
    """
    y = np.load(np_path+"_array_label_rCu.npy")
    _X = np.load(np_path+"_array_spectra.npy")
    spectra_len = (stop_wl_id - start_wl_id)
    X = np.zeros([y.size, ill_pos_count*spectra_len])
    vol_id = np.load(np_path+"_array_vol_id.npy")
    # generate array of L1 norms
    for i in range(y.size):
        L1 = np.zeros(ill_pos_count)
        for ill in range(ill_pos_count):
            L1[ill] = np.sum(_X[ill, start_wl_id:stop_wl_id, i])
        if sort:
            # normalize spectra with their L1 norms 
            # after sorting them by decr L1 norms
            for ill in range(ill_pos_count):
                X[i, ill*spectra_len:(ill+1)*spectra_len] = _X[np.argmax(L1), 
                    start_wl_id:stop_wl_id, i]/L1[np.argmax(L1)]
                L1[np.argmax(L1)] = 0
        else:
            # normalize spectra with their L1 norms 
            for ill in range(ill_pos_count):
                X[i, ill*spectra_len:(ill+1)*spectra_len] = _X[ill, 
                    start_wl_id:stop_wl_id, i]/L1[ill]
    if data_augment:
        y_aug = np.zeros([y.size*2])
        y_aug[:y.size] = y
        y_aug[y.size:y.size*2] = y
        X_aug = np.zeros([y.size*2, ill_pos_count*spectra_len])
        X_aug[:y.size, :] = X
        # include data with mirrored illumination
        for i in range(y.size):
            L1 = np.zeros(ill_pos_count)
            for ill in range(ill_pos_count):
                L1[ill] = np.sum(_X[ill, start_wl_id:stop_wl_id, i])
            # normalize spectra with their L1 norms
            # but mirror them
            for ill in range(ill_pos_count):
                X_aug[i+y.size, ill*spectra_len:(ill+1)*spectra_len] = _X[ill, 
                    start_wl_id:stop_wl_id, i]/L1[ill]
        return y_aug, X_aug  
    return y, X

def preprocess_np_for_lsd(np_path, start_wl_id=0, stop_wl_id=16, 
                          ill_pos_count=4):
    """ Returns numpy arrays of labels y and their feature vectors X,
    as well as the summed signal S over the spectrum. 
    For "normal" learned spectral decoloring (LSD) from multiple
    illumination data. A feature vector X is an L1 normalized mean 
    of the measured multiple illumination spectra. Thereby approximating 
    the spectrum that would be measured by concurent rather then 
    consecutive illumination.
    A custom range of wavelengths can be selected by specifying 
    start_wl_id and stop_wl_id accordingly.
    
    Keyword arguments:
    np_path -- string path to the numpy arrays to be preprocessed
    """
    y = np.load(np_path+"_array_label_rCu.npy")
    _X = np.load(np_path+"_array_spectra.npy")
    spectra_len = (stop_wl_id - start_wl_id)
    X = np.zeros([y.size, spectra_len])
    S = np.zeros([y.size])
    vol_id = np.load(np_path+"_array_vol_id.npy")
    # generate array of L1 norms
    for i in range(y.size):
        X[i, :] = np.sum(_X[:, start_wl_id:stop_wl_id, i], axis=0)
        # normalize spectra with their L1 norms
        S[i] = np.sum(X[i, :])
        X[i, :] = X[i, :]/S[i]
    return y, X, S

def preprocess_df_for_milsd(df, start_wl_id=0, stop_wl_id=16, 
                            ill_pos_count=4, spectrum_len_df=16, sort=True):
    """ Returns numpy arrays of labels y and their feature vectors X; 
    For multiple illumination learned spectral decoloring (MI-LSD).
    The feature vectors X are ill_pos_count of L1 normalized spectra 
    sorted after their decreasing L1 norms.
    A custom range of wavelengths can be selected by specifying 
    start_wl_id and stop_wl_id accordingly.
    
    Keyword arguments:
    df -- pandas dataframe to be preprocessed
    """
    # i'm sure there are much more elegant ways to do this :D
    y = np.asarray(df['label_rCu'])
    spectra_len = (stop_wl_id - start_wl_id)
    X = np.zeros([y.size, ill_pos_count*spectra_len])
    # generate array of L1 norms
    for i in range(y.size):
        s = np.zeros([ill_pos_count,spectra_len])
        L1 = np.zeros(ill_pos_count)
        for ill in range(ill_pos_count):
            _from = ill*spectrum_len_df+start_wl_id
            _to = (ill+1)*spectrum_len_df-(spectrum_len_df-stop_wl_id)
            s[ill, :] = np.squeeze(np.asarray(np.matrix(df['descriptor'][i]))
                                  )[_from:_to]
            # i'm sorry you have to see this
            L1[ill] = np.sum(s[ill, :])
        if sort:
            # normalize spectra with their L1 norms 
            # after sorting them by decr L1 norms
            for ill in range(ill_pos_count):
                X[i, ill*spectra_len:(ill+1)*spectra_len] = (
                    s[np.argmax(L1), :]/L1[np.argmax(L1)])
                L1[np.argmax(L1)] = 0
        else:
            # normalize spectra with their L1 norms 
            for ill in range(ill_pos_count):
                X[i, ill*spectra_len:(ill+1)*spectra_len] = (
                    s[ill, :]/L1[ill])
    return y, X

def preprocess_df_for_lsd(df, start_wl_id=0, stop_wl_id=16, 
                          ill_pos_count=4, spectrum_len_df=16):
    """ Returns numpy arrays of labels y and their feature vectors X,
    as well as the summed signal S over the spectrum. 
    For "normal" learned spectral decoloring (LSD) from multiple
    illumination data. A feature vector X is an L1 normalized mean 
    of the measured multiple illumination spectra. Thereby approximating 
    the spectrum that would be measured by concurent rather then 
    consecutive illumination.
    A custom range of wavelengths can be selected by specifying 
    start_wl_id and stop_wl_id accordingly.
    
    Keyword arguments:
    df -- pandas dataframe to be preprocessed
    """
    y = np.asarray(df['label_rCu'])
    spectra_len = (stop_wl_id - start_wl_id)
    X = np.zeros([y.size, spectra_len])
    S = np.zeros([y.size])
    for i in range(y.size):
        s = np.zeros([ill_pos_count,spectra_len])
        for ill in range(ill_pos_count):
            _from = ill*spectrum_len_df+start_wl_id
            _to = (ill+1)*spectrum_len_df-(spectrum_len_df-stop_wl_id)
            s[ill, :] = np.squeeze(np.asarray(np.matrix(df['descriptor'][i]))
                                  )[_from:_to]
        s_sum = np.sum(s[:, :], axis=0)
        S[i] = np.sum(s_sum)
        X[i,:] = s_sum/S[i]
    return y, X, S
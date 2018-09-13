#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:37:16 2018

@author: apatane
utils used to get dense predictions directly from raw input signals.
"""


import wfdb
import numpy as np
from data_generator import rearrange_channels, pre_processing
import h5py

def get_data_windows(currMatFilePath,win_size_fs,downSampling,overlap,test):
    val, _ = wfdb.rdsamp(currMatFilePath)
    val = np.float32(val)
    curr_user_n_dense_samples = val.shape[0]
    val = val[::downSampling,:]
    if not test:
        with h5py.File(currMatFilePath+'-arousal.mat', 'r') as f:
            arousalLabel = f['data']['arousals']
            arousalLabel = arousalLabel[:]
        arousalLabel = arousalLabel[::downSampling]
    length = len(val)
    startIdx = 0
    finIdx = win_size_fs
    deltaWindow = int(np.floor((1 - overlap)*win_size_fs))

    winCount = 1
    x_temp = []
    y_temp = []
    while finIdx <= length:
        winVal = val[startIdx:finIdx,:]
        x_temp.append(winVal)
        currLabelVec = np.array(arousalLabel[startIdx:finIdx]);
        currLabelVoting = [np.sum(currLabelVec == -1),np.sum(currLabelVec == 0),
                           np.sum(currLabelVec == 1)]
        idx = np.argmax(currLabelVoting);
        idx = idx - 1;
        
        y_temp.append(    idx    )
        winCount = winCount + 1
        startIdx = startIdx + deltaWindow
        finIdx = finIdx + deltaWindow
        
    x_temp = np.array(x_temp)
    return x_temp, curr_user_n_dense_samples, y_temp



def from_win_prediction_to_dense(model_output,win_upsample,overlap,len_y_dense):
    '''
        Uses the prediction of overlapping time windows (model_output) to compute dense prediction (that is @200Hz)
    '''
    dense_pred = np.zeros((len_y_dense,),dtype = np.float16)
    count_pred = np.zeros((len_y_dense,),dtype = np.int16)
    delta = int(np.floor((1.0-overlap)*win_upsample))
    init = 0
    fin = int(win_upsample)
    for i in range(model_output.shape[0]):
        dense_pred[init:fin] = dense_pred[init:fin] + model_output[i]
        count_pred[init:fin] = count_pred[init:fin] + 1 
        init = init + delta
        fin = fin + delta
    
    for i ,count in enumerate(count_pred):
        if count == 0:
            count_pred[i] = 1
            
        
    dense_pred = dense_pred / count_pred
    

    return dense_pred      




def prepare_input_data(x_temp,samplesPerWin,siameseNet,channels2Use,convModelDims,overlap,winSize):
    flattened_channels = [item for sublist in channels2Use for item in sublist]
    x_temp = x_temp[:,:,flattened_channels]
    x_s = np.zeros((len(x_temp),samplesPerWin,len(flattened_channels)),dtype = np.float32)
    if siameseNet >= 1:
        x_s_prev = np.zeros((len(x_temp),samplesPerWin,len(flattened_channels)),dtype = np.float32)
        if siameseNet >= 2:
            x_s_next = np.zeros((len(x_temp),samplesPerWin,len(flattened_channels)),dtype = np.float32)
        else:
            x_s_next = []
    else:
        x_s_prev = []
        x_s_next = []

    for i in range(len(x_temp)):
        x_s[i] = pre_processing(x_temp[i],flattened_channels,samplesPerWin,winSize)
    
    if siameseNet >= 1:
        for i in range(len(x_temp)):
            x_s_prev[i]  = pre_processing(x_temp[max(0,i-int(1./(1-overlap) ) )  ],flattened_channels,samplesPerWin,winSize, normFlag = False)
    if siameseNet == 2:
        for i in range(len(x_temp)):
            x_s_next[i]  = pre_processing(x_temp[min(i+int(1./(1-overlap)),len(x_temp) - 1)],flattened_channels,samplesPerWin,winSize, normFlag = False)
        
    x_s = rearrange_channels(x_s,convModelDims,channels2Use)           
    if siameseNet >= 1:
        x_s_prev = rearrange_channels(x_s_prev,convModelDims,channels2Use)   
    if siameseNet == 2:
        x_s_next = rearrange_channels(x_s_next,convModelDims,channels2Use) 
    return x_s + x_s_prev + x_s_next

def get_raw_data_path():
    raw_data_path = '../data/training/'
    return raw_data_path
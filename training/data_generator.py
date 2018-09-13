#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 09:26:06 2018

@author: __pat__

In this file I define the various utils used by the data generators used at training time
"""

import pickle
import numpy as np
from utils import get_new_indexing
from keras.utils import to_categorical
import os
from freq_comp import comp_inst_phase


def generator(data_path,batch_size,user_ids,chs,n_sub_sampled,convModelDims,siameseNet, y_alls_pos, y_alls_neg,winSize):
'''
This is the actual data generator. Super inefficient as it is...
'''
    
    flattened_channels = [item for sublist in chs for item in sublist]
    while True:
        x_s = np.zeros((batch_size,n_sub_sampled,len(flattened_channels)),dtype = np.float32)
        if siameseNet >= 1:
            x_s_prev = np.zeros((batch_size,n_sub_sampled,len(flattened_channels)),dtype = np.float32)
            if siameseNet >= 2:
                x_s_next = np.zeros((batch_size,n_sub_sampled,len(flattened_channels)),dtype = np.float32)
            else:
                x_s_next = []
        else:
            x_s_prev = []
            x_s_next = []
        y_s = np.zeros((batch_size),dtype = np.int16)

        #the first half of the batch is taken from positive arousal sample
        for i in range(batch_size/2):
            temp_vec = []
            while len(temp_vec) == 0:
                u_idx = np.random.choice(len(user_ids))
                id = user_ids[u_idx]
                temp_vec = y_alls_pos[u_idx]
            rand_idx = np.random.choice(temp_vec)
            y_s[i] = np.int16(1)
            x_temp = np.asarray(pickle.load(open(data_path + id + '/' + 'x_' + str(rand_idx) + '.p')),dtype = np.float32)
            x_temp = x_temp[:,flattened_channels]
            x_s[i] = pre_processing(x_temp,flattened_channels,n_sub_sampled,winSize)
            if siameseNet >= 1:
                prev_file_name = get_prev_file_name(data_path + id +  '/',rand_idx )
                x_temp = np.asarray(pickle.load(open(prev_file_name)),dtype = np.float32)
                x_temp = x_temp[:,flattened_channels]
                x_s_prev[i]  = pre_processing(x_temp,flattened_channels,n_sub_sampled,winSize)                    
                if siameseNet == 2:
                    len_yall = len(y_alls_pos[u_idx]) + len(y_alls_neg[u_idx])
                    next_file_name = get_next_file_name(data_path + id +  '/',rand_idx,len_yall)
                    x_temp = np.asarray(pickle.load(open(next_file_name)),dtype = np.float32)
                    x_temp = x_temp[:,flattened_channels]
                    x_s_next[i] = pre_processing(x_temp,flattened_channels,n_sub_sampled,winSize)
        
        #the second half of the batch is taken from negative arousal sample          
        for i in range(batch_size/2,batch_size):
            u_idx = np.random.choice(len(user_ids))
            id = user_ids[u_idx]
            rand_idx = np.random.choice(y_alls_neg[u_idx])
            y_s[i] = np.int16(0)
            x_temp = np.asarray(pickle.load(open(data_path + id + '/' + 'x_' + str(rand_idx) + '.p')),dtype = np.float32)
            x_temp = x_temp[:,flattened_channels]
            x_s[i] = pre_processing(x_temp,flattened_channels,n_sub_sampled,winSize)
            if siameseNet >= 1:
                prev_file_name = get_prev_file_name(data_path + id +  '/',rand_idx )
                x_temp = np.asarray(pickle.load(open(prev_file_name)),dtype = np.float32)
                x_temp = x_temp[:,flattened_channels]
                x_s_prev[i]  = pre_processing(x_temp,flattened_channels,n_sub_sampled,winSize)                    
                if siameseNet == 2:
                    len_yall = len(y_alls_pos[u_idx]) + len(y_alls_neg[u_idx])
                    next_file_name = get_next_file_name(data_path + id +  '/',rand_idx,len_yall)
                    x_temp = np.asarray(pickle.load(open(next_file_name)),dtype = np.float32)
                    x_temp = x_temp[:,flattened_channels]
                    x_s_next[i]  = pre_processing(x_temp,flattened_channels,n_sub_sampled,winSize)
   
                    
        x_s = rearrange_channels(x_s,convModelDims,chs)           
        
        if siameseNet >= 1:
            x_s_prev = rearrange_channels(x_s_prev,convModelDims,chs)   
            if siameseNet == 2:
                x_s_next = rearrange_channels(x_s_next,convModelDims,chs) 
        
        yield x_s + x_s_prev + x_s_next, to_categorical(y_s,num_classes = 2)
    
   
    
def per_sample_x_axis_sub_sample(x_train,x_axis_size):
    '''
    cropping time signlas for data augmentation
    '''
    diff = x_train.shape[0] - x_axis_size 
    intIdx = np.random.randint(diff)
    finIdx = diff - intIdx
    return x_train[intIdx:(x_train.shape[0]-finIdx),:]

def pre_processing(x_temp,flattened_channels,n_sub_sampled,winSize, normFlag = True):
    #pre-processing of signals. This include standardisation, and frequency computation for EEG
    data_means = [-1.58837489e-04, -1.11909714e-04,  1.85214086e-05,  3.43906861e-06,
                      -4.20280479e-04, -4.93160169e-06,  1.79198553e-05,  7.16557753e-05,
                      -4.05202774e-03, -1.71058490e-01, -1.55950109e-02,  9.34850143e+01,
                      -1.74383175e-06]
    data_std = [3.82136545e+01, 4.03138769e+01, 3.92392071e+01, 4.47130111e+01,
                    3.96919219e+01, 4.08547734e+01, 3.77899386e+01, 5.63194308e+01,
                    6.32337194e+02, 3.99307484e+02, 5.43303006e+02, 8.40357161e+00,
                    2.09212648e-01]
    if normFlag:
        for j in range(x_temp.shape[1]):
            x_temp[:,j] = (x_temp[:,j] - data_means[flattened_channels[j]])/data_std[flattened_channels[j]]
    for j in range(len(flattened_channels)):
        if (flattened_channels[j]>=0) and (flattened_channels[j]<=5):
            x_temp[:,j] = comp_inst_phase(x_temp[:,j])
    if n_sub_sampled < x_temp.shape[0]:
        x_s_i = per_sample_x_axis_sub_sample(x_temp,n_sub_sampled)
    else:
        x_s_i = x_temp
    return x_s_i

def rearrange_channels(x_s,convModelDims,chs):
    '''
    re-arranging channels in a way that is compatible with keras input for training
    '''
    x_s = x_s[:,:,:,np.newaxis]
    x_s = [x_s[:,:,c_s,:] if convModelDims[j] == 2 else np.transpose(x_s[:,:,c_s,:],[0,1,3,2])
            for j, c_s in enumerate(get_new_indexing(chs))]
    return x_s

def get_prev_file_name(preName,rand_idx):
    '''
    utils to get name of previous data sample. This is an ugly function really, it is necessary because of bad choices made at the
    beginning of the coding. It should be removed, and things should get cleaned up a bit
    '''
    if os.path.isfile(preName + 'x_prev_' + str(rand_idx) + '.p'):
        prev_file_name = preName + 'x_prev_' + str(rand_idx) + '.p'
    else:
        prevIdx = max(rand_idx - 1,0)
        prev_file_name = preName + 'x_' + str(prevIdx) + '.p'
    return prev_file_name

def get_next_file_name(preName,rand_idx,len_y_s):
    '''
    same as above but for the successive input sample
    '''
    if os.path.isfile(preName + 'x_next_' + str(rand_idx) + '.p'):
        next_file_name = preName + 'x_next_' + str(rand_idx) + '.p'
    else:
        nextIdx = min(rand_idx + 1,len_y_s - 1)
        next_file_name = preName + 'x_' + str(nextIdx) + '.p'
    return next_file_name



    

    
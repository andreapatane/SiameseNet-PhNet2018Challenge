#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:33:51 2018

@author: apatane
"""
import numpy as np
import pickle



def dataPath_and_cpus():
    '''Define where the training data are, and the maximum number of CPUs that
       keras will use (if not in GPU mode).
    '''
    numOfCPU = 2
    workers = 2
    DATASET_PATH = '../data/'

    return DATASET_PATH, numOfCPU, workers

def get_new_indexing(channels2Use):
    channels2Use_flat = [item for sublist in channels2Use for item in sublist]
    new_indexing = []
    for ch_list in channels2Use: 
        curr_new_indexing = []
        for ch in ch_list:
            curr_new_indexing.append([i for i, c in enumerate(channels2Use_flat) if c == ch][0])
        new_indexing.append(curr_new_indexing)
    
    return new_indexing

def get_num_of_samples(data_dir,userIDs):
    n_samples = 0
    n_pos = 0
    n_neg = 0
    for currUser in userIDs:
        y = pickle.load(open(data_dir + currUser + '/y.p'))
        n_samples = n_samples + len(y)
        n_pos = n_pos + np.sum(   np.asarray(y) == 1   )
        n_neg = n_neg + np.sum(   np.asarray(y) == 0   )
    
    return n_samples, n_pos, n_neg
    

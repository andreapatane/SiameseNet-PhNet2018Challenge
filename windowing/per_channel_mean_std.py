#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 19:04:05 2018

@author: __pat__
Pre-computed mean and standard deviation for each channel. This is then used to standardise every signal
"""

import numpy as np
import pickle
import glob
from utils import dataPath_and_cpus
import os
import time


dataDir , _, _ = dataPath_and_cpus()
numOfChs = 13
dataDir = dataDir +  'cache/non_overlap_' + str(60) + '/'

means = np.zeros((13),dtype = np.float64)
stds = np.zeros((13),dtype = np.float64)


users = os.listdir(dataDir)
users = [u for u in users if u[:2] == 'tr']
numOfUsers = len(users)

for i in range(numOfChs):
    print 'channel: ' + str(i)
    per_user_mean = np.zeros((numOfUsers),dtype = np.float64)
    
    samples_per_user = np.zeros((numOfUsers),dtype = np.float64)
    print 'computing mean:'
    start_time = time.time()
    for j in range(numOfUsers):
        currDir = dataDir + users[j] + '/'
        x_s_names = glob.glob(currDir + 'x_*')
        x_s = []
        for _, name in enumerate(x_s_names):
            x_temp = pickle.load(open(name))
            x_s.append(x_temp[:,i])
        x_s = np.asarray(x_s,dtype = np.float64)
        per_user_mean[j] = np.mean(x_s)
        samples_per_user[j] = np.shape(x_s)[0]
    means[i] = np.dot(per_user_mean,samples_per_user)/np.sum(samples_per_user)
    print 'time elapsed: ' + str(time.time() - start_time)
    
    per_user_var = np.zeros((numOfUsers),dtype = np.float64)
    print 'computing variance:'
    start_time = time.time()
    for j in range(numOfUsers):
        currDir = dataDir + users[j] + '/'
        x_s_names = glob.glob(currDir + 'x_*')
        x_s = []
        for _, name in enumerate(x_s_names):
            x_temp = pickle.load(open(name))
            x_s.append(x_temp[:,i])
        x_s = np.asarray(x_s,dtype = np.float64)
        per_user_var[j] = np.mean((x_s - means[i])**2)
    stds[i] = np.sqrt(np.dot(samples_per_user,per_user_var)/(np.sum(samples_per_user) - 1) )
    print 'time elapsed: ' + str(time.time() - start_time)
        
print means
print stds           
            
                        
    
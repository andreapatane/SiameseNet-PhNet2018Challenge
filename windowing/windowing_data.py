########################################################
#Script for preprocessing of raw data into time windows#
########################################################
import os
import numpy as np
import pickle
import h5py
import time
import wfdb

#raw data directory
fileDir = '../data/training'
#parent of directory where to save windowed data
resDir = '.../data/'


fileList = os.listdir(fileDir)
fileList = [f for f in fileList if f[:2] == 'tr']



windowSize = 30
downSampling = 4 #downsampling ratio for the signal
fs = 200
numOfChannels = 13
win_size_fs = int(round(windowSize*(fs/float(downSampling))))
overlap = False
cumNeg = 0
cumPos = 0

for i in range(0,len(fileList)):
    start_time = time.time()
    print(str(i+1))

    currPatient = fileList[i]
    currSubFolder =  fileDir + currPatient + '/'
    dataFile = currSubFolder + currPatient
    val, _ = wfdb.rdsamp(dataFile)
    val = np.float16(val)
    arousalFile = currSubFolder + currPatient + '-arousal.mat'
    with h5py.File(arousalFile, 'r') as f:
        arousalLabel = f['data']['arousals']
        arousalLabel = arousalLabel[:]
         
        
    val = val[::downSampling,:]


    if overlap:
        templName = 'overlap_'
    else:
        templName = 'non_overlap_'
        
    templName = templName +  str(windowSize)
    
    userCacheDir = resDir + 'cache/' + templName + '/' 
    if not os.path.isdir(userCacheDir):
        os.mkdir(userCacheDir)
    
    userCacheDir = userCacheDir + fileList[i] + '/'
    if not os.path.isdir(userCacheDir):
        os.mkdir(userCacheDir)
        
    
    arousalLabel = arousalLabel[::downSampling]
    
    length = len(val)
    startIdx = 0
    finIdx = win_size_fs
    if overlap:
        deltaWindow = int(np.floor(win_size_fs/2.))
    else:
        deltaWindow = int(win_size_fs)
    
    winCount = 1
    x_temp = []
    y_temp = []
    while finIdx <= length:
        winVal = val[startIdx:finIdx,:]
        currLabelVec = np.array(arousalLabel[startIdx:finIdx]);
        currLabelVoting = [np.sum(currLabelVec == -1),np.sum(currLabelVec == 0),
                           np.sum(currLabelVec == 1)]
        idx = np.argmax(currLabelVoting);
        idx = idx - 1;
        y_temp.append(idx)
        x_temp.append(winVal)
        winCount = winCount + 1
        startIdx = startIdx + deltaWindow
        finIdx = finIdx + deltaWindow
    x_temp = np.array(x_temp,dtype = np.float16 )
    y_aux = []
    valid_count = 0;
    pos = np.sum(np.asarray(y_temp) == 1.)
    neg = np.sum(np.asarray(y_temp) == 0.)
    
    for j in range(np.size(x_temp,0)):
        if y_temp[j] >= 0:
           
            pickle.dump(x_temp[j],open(userCacheDir + 'x_' + str(valid_count) + '.p','w+'))
            
            if j > 0:
                prevIdx = j - 1
            else:
                prevIdx = j
            if y_temp[prevIdx] == -1:
                pickle.dump(x_temp[prevIdx],open(userCacheDir + 'x_prev_' + str(valid_count) + '.p','w+'))
                
            if j < (len(y_temp) - 1):
                nextIdx = j + 1
            else:
                nextIdx = j
            if y_temp[nextIdx] == -1:
                pickle.dump(x_temp[nextIdx],open(userCacheDir + 'x_next_' + str(valid_count) + '.p','w+'))
            
            valid_count = valid_count + 1
            y_aux.append(y_temp[j])
  
    pickle.dump(np.asarray(y_aux,dtype = np.int16),open(userCacheDir + 'y.p','w+'))
    print("--- %s seconds ---" % (time.time() - start_time))

 
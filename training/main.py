#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:30:40 2018

@author: __pat__
Main script file for NN training

"""


from utils import dataPath_and_cpus
from keras.optimizers import Adam
import numpy as np
import keras.backend as K
import tensorflow as tf
from model_utils import merge_conv_models 
from callbacks import ModelTest
import os
import pickle
from data_generator import generator
from utils import get_num_of_samples
from keras.models import model_from_json
import sys

############################################################
###CONFIGURATION STUFF, depend on specifc of machine   #####
############################################################
dataDir , numOfCPU, workers = dataPath_and_cpus()
#setting GPU device to use.
os.environ['CUDA_VISIBLE_DEVICES']="1"

#setting maximum number of cores in tensorflow backend
config = tf.ConfigProto(intra_op_parallelism_threads=numOfCPU, inter_op_parallelism_threads=numOfCPU, \
                        allow_soft_placement=True, device_count = {'CPU': numOfCPU})
session = tf.Session(config=config)
K.set_session(session)




#####################################
###SETTING UP SOME PARAMETERS...#####
#####################################
training2do = True
validation2do = True
testing2do = True
totalNumOfUser = 994
siameseNet = 2
channels2Use = [[3],[6],[8],[10],[11],[12]]
convModelDims = [1,1,1,1,1,1]
typeOfModels = ['new_cnn2','new_cnn2','new_cnn2','new_cnn2','mlp','new_cnn2']
mergingMLP = 'a_bit_deep_wider'
modelID = '3_6_8_10_11_12'
useBatchNorm = False
prelu = True
winSize = 30
data_path = dataDir + 'cache/non_overlap_' + str(winSize) + '/'
test_data_dir = '../data/test/'
batch_size = 32
useGenerator = False
n_ep = 30
samplesPerWin = 1400
valNum = 100
testNum = 100

######################################################
###DEFINING TRAINING, VALIDATION AND TEST SETS...#####
######################################################
users = os.listdir(data_path)
users = [u for u in users if u[:2] == 'tr']
users = users[:totalNumOfUser]
users.sort()

if testing2do:
    test_users = os.listdir(test_data_dir)
    test_users = [u for u in test_users if u[:2] == 'te']
    test_users.sort()





userIDs = users[(valNum+testNum):]
testUserIDs = users[:testNum]
valUserIDs = users[testNum:(testNum+valNum)]

n_samples, n_pos, n_neg = get_num_of_samples(data_path,userIDs)

print 'num of samples: ' + str(n_samples)
print 'Positive to Negative Ratio: ' + str( float(n_pos)/float(n_neg) )


##########################################
###building up architecture blocks...#####
##########################################

smallConvFilters = [4   ,4    ,4   ,8  ,16  ,16]
smallConvKernels = [5   ,5    ,5   ,5  ,3   ,3]
smallConvPool = [False,True,False,True,False,True]

newConvFilters = [4   ,4    ,4   ,8  ,16  ,16]
newConvFilters2 = [8   ,8    ,8   ,16  ,32  ,32]
newConvFilters3 = [8   ,8    ,16   ,16  ,32 ,64]

newConvKernels = [16   ,16    ,16   ,32  ,32   ,64]
newConvKernels3 = [16   ,16    ,16   ,32  ,32   ,64]

newConvPool = [False,True,False,True,False,True]


if mergingMLP == 'simple':
    mergingDenseUnits = []
    mergingDO = []
elif mergingMLP == 'a_bit_deep':
    mergingDenseUnits = [256,256]
    mergingDO = [0.25,0.25]
elif mergingMLP == 'a_bit_deep_wider':
    mergingDenseUnits = [512,512]
    mergingDO = [0.33,0.33]
elif mergingMLP == 'a_bit_deeper_wider':
    mergingDenseUnits = [1024,512,256]
    mergingDO = [0.33,0.33,0.33]

convFilters = []
convKernels = []
convPool = []
convDO = []
for curr_mod in typeOfModels:
    if curr_mod == 'cnn':
        convFilters.append(smallConvFilters)
        convKernels.append(smallConvKernels)
        convPool.append(smallConvPool)
        convDO.append(0.33)
    elif curr_mod == 'new_cnn':
        convFilters.append(newConvFilters)
        convKernels.append(newConvKernels)
        convPool.append(newConvPool)
        convDO.append(0.33)
    elif curr_mod == 'new_cnn2':
        convFilters.append(newConvFilters2)
        convKernels.append(newConvKernels)
        convPool.append(newConvPool)
        convDO.append(0.33)
    elif curr_mod == 'new_cnn3':
        convFilters.append(newConvFilters3)
        convKernels.append(newConvKernels3)
        convPool.append(newConvPool)
        convDO.append(0.33)
    elif curr_mod == 'mlp':
        convFilters.append([])
        convKernels.append([])
        convPool.append([])
        convDO.append(0.33)


rnnUnits = []
rnnDO = []
for curr_mod in typeOfModels:
    rnnUnits.append([])
    rnnDO.append(0.2)


denseUnits = []
denseDO = []

for curr_mod in typeOfModels:
    if curr_mod == 'cnn':
        denseUnits.append([512,256,128])
        denseDO.append([0.33])
    if curr_mod == 'new_cnn':
        denseUnits.append([256,128,128])
        denseDO.append([0.33])
    if curr_mod == 'new_cnn2':
        denseUnits.append([256,128,128])
        denseDO.append([0.33])
    if curr_mod == 'new_cnn3':
        denseUnits.append([256,128,128])
        denseDO.append([0.33])
    elif curr_mod == 'mlp':
        denseUnits.append([512,256,64,64])
        denseDO.append([0.33])

##################################
###MERGING MODELS TOGETHER...#####
##################################

    
input_dims = []
for i in range(len(convModelDims)):
    if convModelDims[i] == 1:
        input_dims.append((samplesPerWin, 1, len(channels2Use[i]) ))
    elif convModelDims[i] == 2:
        input_dims.append((samplesPerWin, len(channels2Use[i]),1 ))

        
model = merge_conv_models(useBatchNorm,input_dims,convModelDims,siameseNet,prelu,
                          convFilters,convKernels,convPool,convDO,denseUnits,denseDO,
                          rnnUnits,rnnDO,mergingDenseUnits,mergingDO)    



model.summary()

optimizer = Adam(lr=0.00005)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])


########################################################
###DEFINING CALLBACKS AND STARTING MODEL FITTING...#####
########################################################

if training2do:
    valSetCallBack = ModelTest(1, batch_size, np.nan, valUserIDs,
                 data_path, samplesPerWin, channels2Use, convModelDims, siameseNet, modelID,useGenerator,winSize)
    testSetCallBack = ModelTest(1, batch_size, np.nan, testUserIDs,
                 data_path, samplesPerWin, channels2Use, convModelDims, siameseNet, modelID,useGenerator,winSize)
    callbacks_list = [valSetCallBack,testSetCallBack]
    
    y_alls = []
    y_alls_pos = []
    y_alls_neg = []
    for currUser in userIDs:
        #y_alls.append(np.int16( pickle.load(open(data_path+ currUser + '/' + 'y.p'))))
        aux = np.int16( pickle.load(open(data_path+ currUser + '/' + 'y.p')))
        y_alls_pos.append(np.nonzero(aux == 1)[0])
        y_alls_neg.append(np.nonzero(aux == 0)[0])
        
        
    #model fitting on data generators
    hist = model.fit_generator(generator(data_path,batch_size,userIDs,channels2Use,samplesPerWin,convModelDims,siameseNet,y_alls_pos ,y_alls_neg,winSize),
                               steps_per_epoch = (n_samples/batch_size), epochs = n_ep, use_multiprocessing = True,
                               callbacks = callbacks_list, workers = workers, max_queue_size = 10)


###################################################
###################################################
###################################################
#Trying out the model on dense prediction settings#
###################################################
###################################################
###################################################

from submission_utils import get_data_windows, from_win_prediction_to_dense, prepare_input_data, get_raw_data_path
from new_scoring_function import Challenge2018Score
import h5py


downSampling = 4
overlap = 0.5
repData = 5


modelPath = data_path + modelID +  '_model'
raw_data_path = get_raw_data_path() 

json_file = open(modelPath + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(modelPath + '.h5')
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

scoreObj = Challenge2018Score()
window_scoreObj = Challenge2018Score()





if testing2do:
    for hhh, curr_user in enumerate(test_users):
        sys.stdout.write("\r" + 'iteration: ' + str(hhh) + ' user: ' + curr_user)
        sys.stdout.flush()
        #print 'iteration: ' + str(hhh) + ' user: ' + curr_user
        currMatFilePath = test_data_dir + curr_user + '/' + curr_user
        y_window = [];
        for _ in range(repData):
            x_temp, curr_user_n_dense_samples, _  = get_data_windows(currMatFilePath,int(round(winSize*(200.0/float(downSampling)))),downSampling,overlap,True)
            x_temp = prepare_input_data(x_temp,samplesPerWin,siameseNet,channels2Use,convModelDims,overlap,winSize)
            if not len(y_window):
                y_window = model.predict(x_temp)
                y_window = y_window[:,1]
            else:
                temp_win = model.predict(x_temp)
                y_window = y_window + temp_win[:,1]
        y_window = y_window/repData
        y_pred = np.round(from_win_prediction_to_dense(y_window, winSize*200,overlap,curr_user_n_dense_samples),3)
        y_pred = y_pred[:,np.newaxis]
        np.savetxt("../annotations/" + curr_user + ".vec", y_pred, fmt='%.3f')




if validation2do:
    
    scoreObj = Challenge2018Score()
    window_scoreObj = Challenge2018Score()
    
    for hhh, curr_user in enumerate(valUserIDs):
        sys.stdout.write("\r" + 'iteration: ' + str(hhh) + ' user: ' + curr_user)
        sys.stdout.flush()
        currMatFilePath = raw_data_path + curr_user + '/' + curr_user
        y_window = [];
        for _ in range(repData):
            x_temp, curr_user_n_dense_samples, y_temp = get_data_windows(currMatFilePath,int(round(winSize*(200.0/float(downSampling)))),downSampling,overlap,False)
            x_temp = prepare_input_data(x_temp,samplesPerWin,siameseNet,channels2Use,convModelDims,overlap,winSize)
            if not len(y_window):
                y_window = model.predict(x_temp)
                y_window = y_window[:,1]
            else:
                temp_win = model.predict(x_temp)
                y_window = y_window + temp_win[:,1]
        y_window = y_window/repData
                
            
    
    
        
        
        y_pred = np.round(from_win_prediction_to_dense(y_window, winSize*200,overlap,curr_user_n_dense_samples),3)
        y_pred = y_pred[:,np.newaxis]
        y_window = y_window[:,np.newaxis]
        arousalFile = currMatFilePath + '-arousal.mat'
        with h5py.File(arousalFile, 'r') as f:
            arousalLabel = f['data']['arousals']
            arousalLabel = arousalLabel[:]
        window_scoreObj.score_record(np.asarray(y_temp)[np.asarray(y_temp) >= 0 ],y_window[np.asarray(y_temp) >= 0 ],curr_user)
        scoreObj.score_record(arousalLabel, y_pred, curr_user)
    print 
    print 'Dense level labelling: '
    auroc = scoreObj.gross_auroc()
    auprc = scoreObj.gross_auprc()
    print 'AUROC: ' + str(auroc) 
    print 'AUPR: ' + str(auprc) 
    
    print 'Window level labelling: '                 
    auroc = window_scoreObj.gross_auroc()
    auprc = window_scoreObj.gross_auprc()
    print 'AUROC: ' + str(auroc) 
    print 'AUPR: ' + str(auprc) 
    
     
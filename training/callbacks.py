#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:30:40 2018

@author: __pat__
# Here I define the ModelTest class as a subclass of the Callback method. This gets called after each training epoch.
It is here used to get AUPR and AUROC statistics during trining on validation and test set

"""


import numpy as np
from keras.callbacks import Callback
import pickle
import time
import sys
from new_scoring_function import Challenge2018Score
from data_generator import per_user_generator, rearrange_channels
from data_generator import pre_processing, get_prev_file_name, get_next_file_name
#from biosppy.signals import ecg
#from scipy.signal import resample
         
class ModelTest(Callback):
    '''
    Definition of callback used for computation of AUC on the valdiation set. Either provide Xt and y_dense (that is the numpy arrays of data)
    or the path to data directory and a list of user ids.
    '''
    def __init__(self, test_every_X_epochs, batch_size, rep_idx, valUserIDs,
                 dataDir, samplesPerWin, channels2Use, convModelDims, siameseNet, modelID, useGenerator,winSize ):
        super(ModelTest, self).__init__()
        
        self.test_every_X_epochs = test_every_X_epochs
        self.batch_size = batch_size
        self.bestAUPR = 0.
        self.AUPRhist = []
        self.rep_idx = rep_idx
        self.valUserIDs = valUserIDs
        self.dataDir = dataDir
        self.samplesPerWin = samplesPerWin
        self.channels2Use = channels2Use
        self.convModelDims = convModelDims
        self.siameseNet = siameseNet
        self.modelID = modelID
        self.y = self.init_y_field()
        self.useGenerator = useGenerator
        self.winSize = winSize
        if not self.useGenerator:
            self.cache_Xt()
            self.Xt = []
            
    #initilise labels for the validation set            
    def init_y_field(self):
        y = []
        for currUser in self.valUserIDs:
            y.append(np.int16( pickle.load(open(self.dataDir + currUser + '/' + 'y.p'))))
        return y
    
    
    # Here, I build up the input data matrix for each user, and I cache it to file for quick RAM menagement.
    # It's an ugly solution, it would probably much better to just define some proper data generator for the validation test as well...
    def cache_Xt(self):

        Xt = []
        
        flattened_channels = [item for sublist in self.channels2Use for item in sublist]
        start_time = time.time()
        
        for i,currUser in enumerate(self.valUserIDs):
            sys.stdout.write("\r" + 'pre-processing for validation user #: ' +  str(i))
            sys.stdout.flush()
            x_s = np.zeros((len(self.y[i]),self.samplesPerWin,len(flattened_channels)),dtype = np.float32)
            if self.siameseNet >= 1:
                x_s_prev = np.zeros((len(self.y[i]),self.samplesPerWin,len(flattened_channels)),dtype = np.float32)
                if self.siameseNet >= 2:
                    x_s_next = np.zeros((len(self.y[i]),self.samplesPerWin,len(flattened_channels)),dtype = np.float32)
                else:
                    x_s_next = []
            else:
                x_s_prev = []
                x_s_next = []
            
            for j in range(len(self.y[i])):
                x_temp = np.asarray(pickle.load(open(self.dataDir + currUser + '/' + 'x_' + str(j) + '.p')),dtype = np.float32)
                x_temp = x_temp[:,flattened_channels]
                x_s[j] = pre_processing(x_temp,flattened_channels,self.samplesPerWin,self.winSize)
                if self.siameseNet >= 1:
                    prev_file_name = get_prev_file_name(self.dataDir + currUser +  '/',j )
                    x_temp = np.asarray(pickle.load(open(prev_file_name)),dtype = np.float32)
                    x_temp = x_temp[:,flattened_channels]
                    x_s_prev[j]  = pre_processing(x_temp,flattened_channels,self.samplesPerWin,self.winSize)
                    if self.siameseNet == 2:
                        next_file_name = get_next_file_name(self.dataDir  + currUser +  '/',j,len(self.y[i]))
                        x_temp = np.asarray(pickle.load(open(next_file_name)),dtype = np.float32)
                        x_temp = x_temp[:,flattened_channels]
                        x_s_next[j]  = pre_processing(x_temp,flattened_channels,self.samplesPerWin,self.winSize)
            x_s = rearrange_channels(x_s,self.convModelDims,self.channels2Use)
            if self.siameseNet >= 1:
                x_s_prev = rearrange_channels(x_s_prev,self.convModelDims,self.channels2Use)   
                if self.siameseNet == 2:
                    x_s_next = rearrange_channels(x_s_next,self.convModelDims,self.channels2Use) 
            Xt.append(x_s + x_s_prev + x_s_next)
        print 
        print 'elapsed time for validation set pre-processing: ' + str(time.time() - start_time)
        with open(self.dataDir + 'cached_Xt.p','w') as f:
            pickle.dump(Xt,f)
        return 
    
    # Simple method that just loads the model that was pickled in the function above.
    def load_Xt(self):
        with open(self.dataDir + 'cached_Xt.p') as f:
            Xt = pickle.load(f)
        return Xt
    
    #method called by keras at the beginning of each epoch. In use it to compute statistics on validation and test set
    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.test_every_X_epochs != 0:
            return
        start_time = time.time()
        
        
        scoreObj = Challenge2018Score() #defining scoring object
        self.Xt = self.load_Xt() #loading cached validation set
        
        #looping on all the validation users
        for i, currUser in enumerate(self.valUserIDs):
            prediction = self.model.predict(self.Xt[i], batch_size=self.batch_size)
            prediction = prediction[:len(self.y[i]),1]
            scoreObj.score_record(self.y[i], prediction, currUser)
            
        self.Xt = []    #emptying the Xt field
        auroc = scoreObj.gross_auroc()
        auprc = scoreObj.gross_auprc()
        self.AUPRhist.append(auprc)
        
        if auprc > self.bestAUPR:
            self.bestAUPR = auprc        
            model_json = self.model.to_json()
            fileName = self.dataDir + self.modelID +  "_model"
            with open(fileName + ".json", "w+") as json_file:
                json_file.write(model_json)
            self.model.save_weights(fileName + ".h5")
            print '---------- Saving model to file, stats are: ----------'

        AUCStr = ("%0.4f" % auroc )
        AUPRStr = ("%0.4f" % auprc )
        print("Win_AUC on validation set " + AUCStr + " AUPR: " + AUPRStr)  
        
        print("Elapsed time: %s s" % (time.time() - start_time))
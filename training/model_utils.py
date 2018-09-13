#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:14:17 2018

@author: apatane
Some wrapper around keras function, used to buld CNN and Siamese models.
Look at "merge_conv_models" for the submission model. Most of the other functions are earlier trials.
"""
from keras.layers import Convolution1D, MaxPooling1D, Reshape, PReLU, Input, Bidirectional, LSTM
from keras.layers import Dense, Dropout,MaxPooling2D, Activation, Flatten, Convolution2D, InputLayer, concatenate
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

def buildModel(input_dim,useBatchNorm):

    reg = l2(0.0001)
    
    model=Sequential()
    model.add(InputLayer(input_shape=( input_dim[0], input_dim[1] , input_dim[2]) ))
    
    #
    #model.add(Convolution2D(filters=16, kernel_size=7,
    #                                    padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    #model.add(BatchNormalization(scale=False))
    #model.add(Dropout(0.33))
    #model.add(Activation('relu'))
  
    #
    model.add(Convolution2D(filters=8, kernel_size=7,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    #model.add(MaxPooling2D((2,1)))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    
    #
    model.add(Convolution2D(filters=16, kernel_size=5,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling2D((2,1))) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    #
    #model.add(Convolution2D(filters=32, kernel_size=5,
    #                                    padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    #model.add(BatchNormalization(scale=False))
    #model.add(Dropout(0.33))
    #model.add(Activation('relu'))
    
    
    
    #
    model.add(Convolution2D(filters=16, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling2D((2,1)))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    

    
    #
    model.add(Convolution2D(filters=32, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling2D((2,2)))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    
    #
    model.add(Convolution2D(filters=64, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling2D((2,1)))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    #
    model.add(Convolution2D(filters=128, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling2D((2,1)))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    #
    model.add(Convolution2D(filters=256, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling2D((2,1)))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    #
    model.add(Convolution2D(filters=512, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    #model.add(MaxPooling2D((2,1)))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    #
    model.add(Flatten())
    #
    
    #
    model.add(Dense(units=1024,kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    #
    model.add(Dense(units=256,kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    

    
    
    model.add(Dense(units=2,kernel_regularizer=reg,bias_regularizer=reg))
    model.add(Activation('softmax'))
    return model

def convModel_D1(input_dim,useBatchNorm,prelu):
    
    reg = l2(0.00001)
    do = 0.2
    model=Sequential()
    model.add(InputLayer(input_shape=( input_dim[0] , input_dim[1], input_dim[2]  ) ))
    model.add(Reshape((input_dim[0], input_dim[2]  ) ))

    
    model.add(Convolution1D(filters=4, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    model.add(Convolution1D(filters=4, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D()) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    
    
    
    model.add(Convolution1D(filters=8, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    model.add(Convolution1D(filters=8, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D()) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    
    
    
    
    
    
    model.add(Convolution1D(filters=16, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    model.add(Convolution1D(filters=16, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D()) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    
    
    
    model.add(Convolution1D(filters=32, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    
    model.add(Convolution1D(filters=32, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D()) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    
    
    
    
    
    model.add(Convolution1D(filters=64, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))

    model.add(Convolution1D(filters=64, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D()) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    
    
    model.add(Convolution1D(filters=64, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    model.add(Convolution1D(filters=64, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D()) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    
    
    #
    model.add(Flatten())
    #
    
    #
    model.add(Dense(units=1024,kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
 
    #
    model.add(Dense(units=512,kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
    return model


def convModel_rnn_D1_small(input_dim,useBatchNorm,prelu):
    
    reg = l2(0.00001)
    do = 0.0
    model=Sequential()
    model.add(InputLayer(input_shape=( input_dim[0] , input_dim[1], input_dim[2]  ) ))
    model.add(Reshape((input_dim[0], input_dim[2]  ) ))

    
    model.add(Convolution1D(filters=8, kernel_size=7,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('relu'))
    model.add(Dropout(do))

    
    model.add(Convolution1D(filters=16, kernel_size=5,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D()) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('relu'))
    model.add(Dropout(do))
    

    model.add(Convolution1D(filters=32, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('relu'))
    model.add(Dropout(do))
    
   
    
    #
    model.add(Convolution1D(filters=64, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    #model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('relu'))
    model.add(Dropout(do))

    
    model.add(Bidirectional(LSTM(units=16,return_sequences=True,
                                             dropout=do,kernel_regularizer=reg, recurrent_regularizer=reg  )))
    model.add(Bidirectional(LSTM(units=32,return_sequences=False,
                                             dropout=do,kernel_regularizer=reg, recurrent_regularizer=reg  )))

    
  
    return model


def convModel_D1_small(input_dim,useBatchNorm,prelu):
    
    reg = l2(0.00001)
    do = 0.15
    model=Sequential()
    model.add(InputLayer(input_shape=( input_dim[0] , input_dim[1], input_dim[2]  ) ))
    model.add(Reshape((input_dim[0], input_dim[2]  ) ))

    model.add(Convolution1D(filters=4, kernel_size=5,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))

    
    model.add(Convolution1D(filters=8, kernel_size=5,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D()) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    

    model.add(Convolution1D(filters=16, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))
    
   
    
    #
    model.add(Convolution1D(filters=32, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))

    
    #
    model.add(Flatten())
    #
    
    #
    model.add(Dense(units=128,kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        if prelu:
            model.add(PReLU(shared_axes=[1]))
        else:
            model.add(Activation('tanh'))
    model.add(Dropout(do))

    return model

def CRNN_1d(input_dim,useBatchNorm):
    
    reg = l2(0.0001)
    
    model=Sequential()
    model.add(InputLayer(input_shape=( input_dim[0] , input_dim[1], input_dim[2]  ) ))
    model.add(Reshape((input_dim[0], input_dim[2]  ) ))


    model.add(Convolution1D(filters=16, kernel_size=7,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    #model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))

    
    #
    model.add(Convolution1D(filters=16, kernel_size=5,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    #model.add(MaxPooling1D()) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    #
    model.add(Convolution1D(filters=32, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    #
    model.add(Convolution1D(filters=32, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    #model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    
    #
    model.add(Convolution1D(filters=64, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    #model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    model.add(Bidirectional(LSTM(units=64,return_sequences=True,
                                             dropout=0.4,kernel_regularizer=reg, recurrent_regularizer=reg  )))
    model.add(Bidirectional(LSTM(units=128,return_sequences=True,
                                             dropout=0.4,kernel_regularizer=reg, recurrent_regularizer=reg  )))
    model.add(Bidirectional(LSTM(units=64,return_sequences=False,
                                             dropout=0.4,kernel_regularizer=reg, recurrent_regularizer=reg  )))

    
    return model


def RNN_1d(input_dim,useBatchNorm):
    
    reg = l2(0.0001)
    
    model=Sequential()
    
    model.add(InputLayer(input_shape=( input_dim[0] , input_dim[1], input_dim[2]  ) ))
    model.add(Reshape((input_dim[0], input_dim[2]  ) ))
    
    model.add(Convolution1D(filters=input_dim[2], kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    model.add(Bidirectional(LSTM(units=16,return_sequences=True,
                                             dropout=0.1,kernel_regularizer=reg, recurrent_regularizer=reg  )))
    model.add(Bidirectional(LSTM(units=32,return_sequences=True,
                                             dropout=0.1,kernel_regularizer=reg, recurrent_regularizer=reg  )))

    model.add(Bidirectional(LSTM(units=16,return_sequences=False,
                                             dropout=0.1,kernel_regularizer=reg, recurrent_regularizer=reg  )))
    return model

def default_model(input_dim,useBatchNorm):
    
    reg = l2(0.0001)
    
    model=Sequential()
    model.add(InputLayer(input_shape=( input_dim[0] , input_dim[1], input_dim[2]  ) ))
    model.add(Reshape((input_dim[0], input_dim[2]  ) ))

    
  
    model.add(Convolution1D(filters=8, kernel_size=7,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))

    
    #
    model.add(Convolution1D(filters=16, kernel_size=5,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D()) 
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
   
    
    
    #
    model.add(Convolution1D(filters=32, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    

    
    #
    model.add(Convolution1D(filters=32, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    
    #
    model.add(Convolution1D(filters=64, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    #
    model.add(Convolution1D(filters=64, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    #
    model.add(Convolution1D(filters=128, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    model.add(MaxPooling1D())
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    #
    model.add(Convolution1D(filters=256, kernel_size=3,
                                        padding='same',kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    #
    model.add(Flatten())
    #
    
    #
    model.add(Dense(units=512,kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
    
    #
    model.add(Dense(units=128,kernel_regularizer=reg,bias_regularizer=reg))
    if useBatchNorm:
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))
    else:
        model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.4))
    
 
    return model


def shallow_model(input_dim,useBatchNorm,convFilters,convKernels,convPool,convDO,denseUnits,denseDO,rnnUnits,rnnDO,convModelDim):
    
    reg = l2(0.00001)
    model=Sequential()
    model.add(InputLayer(input_shape=( input_dim[0] , input_dim[1], input_dim[2]  ) ))
    if convModelDim == 1:
        model.add(Reshape((input_dim[0], input_dim[2]  ) ))

      
    #Convolutional layers block:
    for i in range(len(convFilters)):
        if convModelDim == 1:
            model.add(Convolution1D(filters=convFilters[i], kernel_size=convKernels[i],
                                            padding='same',kernel_regularizer=reg,bias_regularizer=reg))
        else:
            model.add(Convolution2D(filters=convFilters[i], kernel_size=convKernels[i],
                                            padding='same',kernel_regularizer=reg,bias_regularizer=reg))
        if convPool[i]:
            if convModelDim == 1:
                model.add(MaxPooling1D())
            else:
                model.add(MaxPooling2D())
        if useBatchNorm:
            model.add(BatchNormalization(scale=False))
            model.add(Activation('relu'))
        else:
            model.add(PReLU(shared_axes=[1]))
        model.add(Dropout(convDO))

    ret = True
    for i in range(len(rnnUnits)):
        if i == (len(rnnUnits) -1):
            ret = False            
        model.add(Bidirectional(LSTM(units=rnnUnits[i],return_sequences=ret,
                                             dropout=rnnDO,kernel_regularizer=reg, recurrent_regularizer=reg  )))

    
    #flattening up filters
    if ret:
        model.add(Flatten())

    #
    for i in range(len(denseUnits)):
        model.add(Dense(units=denseUnits[i],kernel_regularizer=reg,bias_regularizer=reg))
        if useBatchNorm:
            model.add(BatchNormalization(scale=False))
            model.add(Activation('relu'))
        else:
            model.add(PReLU(shared_axes=[1]))
        model.add(Dropout(denseDO[min(i,len(denseDO) - 1)]))
    
    
    return model

def merge_conv_models(useBatchNorm,input_dims,convModelDims,siameseNet,prelu,convFilters,convKernels,convPool,convDO,
                      denseUnits,denseDO,rnnUnits,rnnDO,mergingDenseUnits,mergingDO):
    model_inputs = []
    model_fun = []
    curr_models = []
    for i in range(len(input_dims)):
        curr_models.append(shallow_model(input_dims[i],useBatchNorm,convFilters[i],convKernels[i],convPool[i],
                                             convDO[i],denseUnits[i],denseDO[i],rnnUnits[i],rnnDO[i],convModelDims[i]))
        model_inputs.append(Input(input_dims[i]))
        model_fun.append(curr_models[i](model_inputs[i]))
        
    for _ in range(siameseNet):        
        for i in range(len(input_dims)):
            model_inputs.append(Input(input_dims[i]))
            model_fun.append(curr_models[i](model_inputs[-1]))
    if len(model_fun) > 1:
        merged = concatenate(model_fun)
        
    else:
        merged = model_fun[0]
        
        
    for i in range(len(mergingDenseUnits)):
        merged = Dense(units=mergingDenseUnits[i],kernel_regularizer=l2(0.00001),bias_regularizer=l2(0.00001))(merged)
        merged = PReLU(shared_axes=[1])(merged)
        merged = Dropout(mergingDO[i])(merged)
    pred_layer = Dense(units=2,kernel_regularizer=l2(0.00001),bias_regularizer=l2(0.00001))(merged)
    pred_layer = Activation('softmax')(pred_layer)
    model = Model(input=model_inputs,output=pred_layer)
    return model
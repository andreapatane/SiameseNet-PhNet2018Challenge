#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 17:28:31 2018

@author: apatane

computing instantaneous frequency for EEG, using methods from:
    Valenza, Gaetano et al., "EEG oscillations during caress-like affective haptic elicitation", 2018
"""
import numpy as np
from scipy.signal import hilbert

def comp_inst_phase(x_temp):
    fs = 50.0    #carefull! sampling frequency is hard coded here...
    analytic_signal = hilbert(x_temp)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi) * fs)
    instantaneous_frequency = np.append(instantaneous_frequency,instantaneous_frequency[-1])
    return instantaneous_frequency

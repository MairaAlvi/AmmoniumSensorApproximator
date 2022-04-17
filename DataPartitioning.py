
from Utility.PreProcessing import DataPreProcessing

import json
import os

import matplotlib.pyplot as plt
import numpy  as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import pandas as pd
import random


base_path = ('/home/maira/Data/UASB_HRAP_Modelling')


# Change this depending on where the files are located on your machine
result_base_path = '/home/maira/github/ResultsJournalSoftSensor'
for seed in [1390,456,377,123,42,786,589]:

    for flag in ["CV1","CV2","CV3","CV4","CV5"]:
        start = None

        data_path = os.path.join(base_path, 'Data', 'Dynamic Simulation')

        effluent = pd.read_csv(os.path.join(data_path,'EffluentWSmergedNoiseFilter.csv'),encoding='latin1')

        time = pd.read_csv(os.path.join(data_path,'time.csv'))
        time.columns = ['time']
        time =time[:effluent.shape[0]]

        effluent['Date'] = effluent['Date'].str.extract('(\d+/\d+/\d+)')
        effluent['Date'] = pd.to_datetime(effluent['Date'], format='%d/%m/%Y')
        effluent['Date'] = pd.DatetimeIndex(effluent['Date']).date
        #

        uniqueDates = effluent['Date'].unique()
        numberOfSamplesInEachDay = np.zeros(len(uniqueDates))

        effluent = pd.concat([time, effluent], axis=1)
        effluent = DataPreProcessing.normalize(effluent)
        windowSize = 24  # window 24 : means 6 hours
        continousDataPointsWindows = effluent.shape[0] // windowSize

        continousWindows = dict()
        chunkStart = 0
        chunkEnd = windowSize
        tempDf = pd.DataFrame()
        for i in range(continousDataPointsWindows):
            tempDf = effluent[chunkStart:chunkEnd]
            chunkStart = chunkEnd
            chunkEnd += windowSize
            dictKey = "w" + str(i)
            continousWindows[dictKey] = tempDf

        random.seed(seed)
        lsOfKeys = list(continousWindows.keys())
        random.shuffle(lsOfKeys)

        sensorVarialbes = ['time', 'pH', 'Turbidity (NTU)', 'rain accumulation (max)', 'D.O. (mg/L)', 'Temperature']
        primaryVariable = ['NH4-N (mg/L)']
        X = None
        yGroundTruth = pd.DataFrame()
        laggedObservation = 4  # 4 === 1 hour

        ls = list(continousWindows.keys())

        randomTestWindows,randomValidationWindow = DataPreProcessing.getFold(flag,lsOfKeys)


        for key in ls:
            if (key in randomValidationWindow) or (key in randomTestWindows):
                continue
            samples = continousWindows[key]
            y = samples[primaryVariable][laggedObservation:]
            yGroundTruth = pd.concat([yGroundTruth, y], axis=0)  # Training data

            samples = samples[sensorVarialbes]
            temp = np.array([samples[i:i + laggedObservation] for i in range(samples.shape[0] - laggedObservation)])
            if X is None:
                X = temp
            else:
                X = np.concatenate([X, temp], axis=0)  # training data

        X_valid = None
        yValidtGroundTruth = pd.DataFrame()
        for key in randomValidationWindow:
            samples = continousWindows[key]
            yValid = samples[primaryVariable][laggedObservation:]
            yValidtGroundTruth = pd.concat([yValidtGroundTruth, yValid], axis=0)

            samples = samples[sensorVarialbes]
            temp = np.array([samples[i:i + laggedObservation] for i in range(samples.shape[0] - laggedObservation)])
            if X_valid is None:
                X_valid = temp
            else:
                X_valid = np.concatenate([X_valid, temp], axis=0)

        X_test = None
        yTestGroundTruth = pd.DataFrame()

        for key in randomTestWindows:
            samples = continousWindows[key]
            yTest = samples[primaryVariable][laggedObservation:]
            yTestGroundTruth = pd.concat([yTestGroundTruth, yTest], axis=0)

            samples = samples[sensorVarialbes]
            temp = np.array([samples[i:i + laggedObservation] for i in range(samples.shape[0] - laggedObservation)])
            if X_test is None:
                X_test = temp
            else:
                X_test = np.concatenate([X_test, temp], axis=0)


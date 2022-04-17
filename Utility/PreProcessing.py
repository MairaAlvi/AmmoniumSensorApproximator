import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

class DataPreProcessing():

    @staticmethod
    def normalize(df):# Scaled data between 0 and 1
        result = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    @staticmethod
    def standardise(df):
        scaled_features =  StandardScaler().fit_transform(df)
        scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
        return scaled_features_df
    @staticmethod
    def getFold(flag,lsOfKeys):
        if flag=="CV1":
            randomValidationWindow = lsOfKeys[:107]  #  %10 percent windows

            randomTestWindows = lsOfKeys[107:214]  #  % 10% percent windows
        elif flag=="CV2":
           randomTestWindows = lsOfKeys[:107]  # %10 percent windows

           randomValidationWindow = lsOfKeys[107:214]  # % 10% percent windows
        elif flag=='CV3':

            randomTestWindows = lsOfKeys[861:861 + 107]  # %10 percent windows

            randomValidationWindow = lsOfKeys[861+107:]  # % 10% percent windows
        elif flag== "CV4":
            randomTestWindows = lsOfKeys[361:361 + 107]  # %10 percent windows

            randomValidationWindow = lsOfKeys[100:100+107]  # % 10% percent windows
        else:
            randomTestWindows = lsOfKeys[200:200 + 107]  # %10 percent windows

            randomValidationWindow = lsOfKeys[600:600 + 107]  # % 10% percent windows
        return randomTestWindows,randomValidationWindow
'''
    Interface for the Trendformer model.
'''

# Imports
import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from trendformer.trendformer import transformer_model, features_to_keep

class TrendformerDriver:
    _instance = None

    def __init__(self):
        if TrendformerDriver._instance:
            self = TrendformerDriver._instance
        else:
            self.model = transformer_model()
            self.model.load_weights("./trendformer/model_weights/10_EMA_3_pred.h5")
            TrendformerDriver._instance = self
        print(self.model.summary())

    # Takes 20 days of data, normalized each column, runs the model and returns the prediction
    def predict(self, data):
        # Normalize the data
        normalized_data = self._normalize(data)

        columns_to_drop = [column for column in normalized_data.columns if column not in features_to_keep]

        normalized_data = normalized_data.drop(columns=columns_to_drop)

        # Convert the DataFrame to a numpy array
        normalized_data_array = normalized_data.values

        # Reshape the array to match the expected input shape
        # normalized_data_array = normalized_data_array.reshape(-1, 20, 29)

        # Add an extra dimension to the array
        normalized_data_array = np.expand_dims(normalized_data_array, axis=0)

        prediction = self.model.predict(normalized_data_array)

        return prediction[0][0]

    
    def _normalize(self, data):
        with open("./trendformer/normalization_parameters/10_EMA_3_pred.json", "r") as file:
            params = json.load(file)

        normalized_data = data.copy()
        
        for column in data.columns:
            mean = params[column]["mean"]
            std = params[column]["std"]
            normalized_data[column] = (data[column] - mean) / std

        return normalized_data

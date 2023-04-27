import numpy as np

def compute_dsdf(drought_data, severity_threshold, duration_thresholds):
    """Compute the Drought Severity Duration Frequency (DSDF) for a given set of drought data, 
        severity threshold, and duration thresholds.
    Args:
        drought_data (ndarray): a list or array of drought severity values
        severity_threshold (str): a float representing the severity threshold (e.g., 0.5 for moderate drought)
        duration_thresholds (ndarray): a list or array of duration thresholds in days (e.g., [7, 14, 30, 60])

    Returns:
        ndarray: A 2D numpy array containing the DSDF values for each duration threshold and return period
    """    
    # Sort the drought data in descending order
    sorted_data = np.sort(drought_data)[::-1]
    
    # Compute the number of years in the drought data
    num_years = len(drought_data) / 365
    
    # Initialize an empty 2D array to hold the DSDF values
    dsdf = np.empty((len(duration_thresholds), len(return_periods)))
    
    # Loop over each duration threshold
    for i, duration in enumerate(duration_thresholds):
        # Loop over each return period
        for j, period in enumerate(return_periods):
            # Compute the number of drought events exceeding the severity threshold for the given duration threshold
            num_events = len([x for x in sorted_data if x >= severity_threshold * duration])
            
            # Compute the probability of a drought event exceeding the severity threshold for the given duration threshold
            prob_event = num_events / num_years
            
            # Compute the DSDF value for the given duration threshold and return period
            dsdf[i, j] = (1 - prob_event) ** period
            
    return dsdf

import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor

def predict_rainfall_3d(rainfall_data, temp_data, humidity_data, prediction_data):
    """
    Predict 3D netCDF rainfall with independent variables like temperature and humidity netCDF.
    
    Args:
        rainfall_data (ndarray): xarray Dataset of historical rainfall data
        temp_data (str): xarray Dataset of historical temperature data
        humidity_data (ndarray): xarray Dataset of historical humidity data
        prediction_data (ndarray): xarray Dataset of input data to predict rainfall
    
    Returns:
        ndarray: A 3D xarray Dataset containing the predicted rainfall values for the input data
    """
    # Extract the input and output variables from the rainfall data
    X = np.stack([temp_data.values.flatten(), humidity_data.values.flatten()], axis=1)
    y = rainfall_data.values.flatten()
    
    # Train a Random Forest model on the historical data
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X, y)
    
    # Make predictions on the input data
    temp_pred = prediction_data['temperature'].values.flatten()
    humidity_pred = prediction_data['humidity'].values.flatten()
    X_pred = np.stack([temp_pred, humidity_pred], axis=1)
    rainfall_pred = rf.predict(X_pred)
    rainfall_pred = np.reshape(rainfall_pred, prediction_data['rainfall'].shape)
    
    # Convert the predicted rainfall values to an xarray Dataset
    dims = prediction_data['rainfall'].dims
    coords = prediction_data['rainfall'].coords
    rainfall_pred = xr.DataArray(rainfall_pred, dims=dims, coords=coords)
    rainfall_pred = xr.Dataset({'rainfall': rainfall_pred})
    
    return rainfall_pred

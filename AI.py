#!/usr/bin/env python
import sys
import joblib
import numpy as np
import pickle
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Load the Random Forest model from a pickle file."""
    try:
        model = joblib.load(model_path)
        logger.debug("Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def load_preprocessing(preproc_path):
    """Load preprocessing information from a pickle file."""
    try:
        with open(preproc_path, 'rb') as f:
            preprocessing_data = pickle.load(f)
        scaler = preprocessing_data['scaler']
        y_mean = preprocessing_data['y_mean']
        y_std = preprocessing_data['y_std']
        logger.debug("Preprocessing info loaded successfully.")
        return scaler, y_mean, y_std
    except Exception as e:
        raise RuntimeError(f"Error loading preprocessing info: {e}")

def predict_single(input_values, scaler, rf_model, y_mean, y_std):
    """
    Predict a single output using the provided model and preprocessing.
    Expects input_values as an array-like with the required features.
    """
    # Ensure input is a numpy array with float type
    input_array = np.array(input_values, dtype=float)
    
    # Apply log1p transformation for the AADT value (assumed to be at index 7)
    try:
        input_array[7] = np.log1p(input_array[7])
    except Exception as e:
        raise ValueError(f"Error applying log1p transformation on AADT value {input_array[7]}: {e}")
    
    # Scale the input data
    input_scaled = scaler.transform(input_array.reshape(1, -1))
    
    # Make a prediction using the Random Forest model
    prediction = rf_model.predict(input_scaled)
    
    # Unstandardize the prediction
    prediction = prediction * y_std + y_mean
    return prediction[0]

def predict_from_gpkg(input_gpkg_path, output_gpkg_path, scaler, rf_model, y_mean, y_std):
    """
    Load data from a geopackage file, perform predictions, and save the results 
    to another geopackage file.
    
    Expects the geopackage file to have a layer with the following columns:
      ['DEMOGIDX_5', 'PEOPCOLORPCT', 'UNEMPPCT', 'pct_residential', 
       'pct_industrial', 'pct_retail', 'pct_commercial', 'AADT',
       'Commute_TripMiles_TripStart_avg', 'Commute_TripMiles_TripEnd_avg']
    """
    try:
        gdf = gpd.read_file(input_gpkg_path)
        logger.debug(f"Original GeoDataFrame loaded with {len(gdf)} records.")
    except Exception as e:
        raise RuntimeError(f"Error reading geopackage file: {e}")
    
    # Preserve an original identifier for tracing
    # Preserve the original 'id' column.
    if 'id' not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf['id'] = gdf.index.astype(str)

    # Define required columns (order matters: index 7 is AADT)
    required_columns = [
        'DEMOGIDX_5', 'PEOPCOLORPCT', 'UNEMPPCT', 'pct_residential', 
        'pct_industrial', 'pct_retail', 'pct_commercial', 'AADT',
        'Commute_TripMiles_TripStart_avg', 'Commute_TripMiles_TripEnd_avg'
    ]
    
    # Log rows with missing data in required columns before filling
    missing_info = gdf[gdf[required_columns].isna().any(axis=1)]
    if not missing_info.empty:
        logger.debug("Rows with missing required data before filling:")
        logger.debug(missing_info[['orig_id'] + required_columns])
    
    # Instead of dropping, fill missing values in the required columns with 0
    fill_defaults = {col: 0 for col in required_columns}
    gdf[required_columns] = gdf[required_columns].fillna(fill_defaults)
    
    logger.debug(f"GeoDataFrame after filling missing values has {len(gdf)} records.")
    
    predictions = []
    for idx, row in gdf.iterrows():
        try:
            feature_values = row[required_columns].values
            pred = predict_single(feature_values, scaler, rf_model, y_mean, y_std)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            predictions.append(None)
    
    gdf['Prediction'] = predictions
    logger.debug("Predictions added. Sample predictions:")
    logger.debug(gdf[['id', 'Prediction']].head())
    
    try:
        gdf.to_file(output_gpkg_path, driver="GPKG")
        logger.debug(f"Predictions saved to '{output_gpkg_path}'.")
    except Exception as e:
        raise RuntimeError(f"Error saving predictions to geopackage: {e}")

def main():
    # Default file paths (if no arguments are provided)
    default_model_path = './AI/RandomForestIsoModel.pkl'
    default_preproc_path = './AI/preprocessing_info.pkl'
    default_input_gpkg = './AI/Large_DataSet2.25.gpkg'
    default_output_gpkg = './AI/Large_DataSet2.25_with_predictions.gpkg'
    
    # If command-line arguments are provided, override the defaults.
    if len(sys.argv) > 1:
        input_gpkg_path = sys.argv[1]
        if len(sys.argv) > 2:
            output_gpkg_path = sys.argv[2]
        else:
            output_gpkg_path = input_gpkg_path.replace(".gpkg", "_with_predictions.gpkg")
    else:
        input_gpkg_path = default_input_gpkg
        output_gpkg_path = default_output_gpkg

    logger.debug(f"Using input geopackage: {input_gpkg_path}")
    logger.debug(f"Output geopackage will be: {output_gpkg_path}")
    
    # Load model and preprocessing data
    rf_model = load_model(default_model_path)
    scaler, y_mean, y_std = load_preprocessing(default_preproc_path)
    
    # Run predictions from the geopackage
    predict_from_gpkg(input_gpkg_path, output_gpkg_path, scaler, rf_model, y_mean, y_std)

if __name__ == '__main__':
    main()

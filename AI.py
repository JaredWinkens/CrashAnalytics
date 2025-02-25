import joblib
import numpy as np
import pickle
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """Load the Random Forest model from a pickle file."""
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
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
        print("Preprocessing info loaded successfully.")
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
    except Exception as e:
        raise RuntimeError(f"Error reading geopackage file: {e}")
    
    # Clean data: remove rows with missing values and duplicates
    gdf = gdf.dropna().drop_duplicates()
    
    # Define required columns (order matters since index 7 is AADT)
    required_columns = [
        'DEMOGIDX_5', 'PEOPCOLORPCT', 'UNEMPPCT', 'pct_residential', 
        'pct_industrial', 'pct_retail', 'pct_commercial', 'AADT',
        'Commute_TripMiles_TripStart_avg', 'Commute_TripMiles_TripEnd_avg'
    ]
    
    missing_cols = [col for col in required_columns if col not in gdf.columns]
    if missing_cols:
        raise RuntimeError(f"The following required columns are missing in the geopackage: {missing_cols}")
    
    predictions = []
    for idx, row in gdf.iterrows():
        try:
            # Extract the feature values in the expected order
            feature_values = row[required_columns].values
            pred = predict_single(feature_values, scaler, rf_model, y_mean, y_std)
            predictions.append(pred)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            predictions.append(None)
    
    # Add predictions as a new column to the GeoDataFrame
    gdf['Prediction'] = predictions
    
    try:
        # Save the updated GeoDataFrame to a new geopackage file
        gdf.to_file(output_gpkg_path, driver="GPKG")
        print(f"Predictions saved to '{output_gpkg_path}'.")
    except Exception as e:
        raise RuntimeError(f"Error saving predictions to geopackage: {e}")

def main():
    # File paths relative to the main directory
    model_path = './AI/RandomForestIsoModel.pkl'
    preproc_path = './AI/preprocessing_info.pkl'
    input_gpkg_path = './AI/Large_DataSet2.25.gpkg'
    output_gpkg_path = './AI/Large_DataSet2.25_with_predictions.gpkg'
    
    # Load model and preprocessing data
    rf_model = load_model(model_path)
    scaler, y_mean, y_std = load_preprocessing(preproc_path)
    
    # Run predictions from the geopackage
    predict_from_gpkg(input_gpkg_path, output_gpkg_path, scaler, rf_model, y_mean, y_std)

if __name__ == '__main__':
    main()

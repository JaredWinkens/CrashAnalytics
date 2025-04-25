#!/usr/bin/env python
import sys
import joblib
import numpy as np
import geopandas as gpd
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_gwr_results(path):
    """Load a pickled mgwr.gwr.GWRResults object."""
    try:
        results = joblib.load(path)
        logger.debug(f"Loaded GWRResults from {path}")
        return results
    except Exception as e:
        raise RuntimeError(f"Error loading GWR results: {e}")


def predict_from_gpkg(input_gpkg, output_gpkg, gwr_results):
    """
    Read input GeoPackage, compute GWR predictions, and write out a new
    GeoPackage with a 'GWR_Prediction' column.
    Assumes gwr_results.predy exists.
    """
    # 1) Load data
    gdf = gpd.read_file(input_gpkg)
    logger.debug(f"Loaded {len(gdf)} features from {input_gpkg}")

    # 2) Ensure an 'id' column for feature matching
    if 'id' not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf['id'] = gdf.index.astype(str)

    # 3) Build coordinates array from centroids
    coords = np.column_stack([
        gdf.geometry.centroid.x.values,
        gdf.geometry.centroid.y.values
    ])

    # 4) Extract the same features used in training
    features = [
        'DEMOGIDX_5', 'PEOPCOLORPCT', 'UNEMPPCT', 'pct_residential',
        'pct_industrial', 'pct_retail', 'pct_commercial', 'AADT',
        'AvgCommuteMiles(Start)', 'AvgCommuteMiles(End)'
    ]
    # Ensure columns exist and are numeric
    for col in features:
        if col not in gdf.columns:
            gdf[col] = 0.0
    X = gdf[features].apply(pd.to_numeric, errors='coerce').fillna(0.0).values

    # 5) Get predictions from the fitted results
    if hasattr(gwr_results, 'predy'):
        preds = gwr_results.predy.flatten().tolist()
    else:
        # fallback if an unfitted model was loaded
        res = gwr_results.predict(coords, X)
        preds = res.predy.flatten().tolist()

    # 6) Align prediction length to dataframe length
    if len(preds) != len(gdf):
        logger.warning(
            f"Prediction count ({len(preds)}) does not match number of features ({len(gdf)}). "
            "Padding with None for missing entries."
        )
        # Pad with None to match
        preds.extend([None] * (len(gdf) - len(preds)))
    # Convert NaN to None for true NULLs
    preds = [None if (p is None or (isinstance(p, float) and np.isnan(p))) else p for p in preds]

    # 7) Attach to GeoDataFrame
    gdf['GWR_Prediction'] = preds

    # 8) Write out to a new GeoPackage
    gdf.to_file(output_gpkg, driver="GPKG")
    logger.debug(f"Wrote GWR predictions to {output_gpkg}")


def main():
    default_results = './AI/gwr_results.pkl'
    default_input = './AI/Rename_DataSet2.25.gpkg'
    default_output = './AI/Rename_DataSet2.25_with_gwr_predictions.gpkg'

    results_path = sys.argv[1] if len(sys.argv) > 1 else default_results
    input_gpkg   = sys.argv[2] if len(sys.argv) > 2 else default_input
    output_gpkg      = sys.argv[3] if len(sys.argv) > 3 else default_output

    logger.debug(f"Using GWR results: {results_path}")
    logger.debug(f"Input GeoPackage: {input_gpkg}")
    logger.debug(f"Output GeoPackage: {output_gpkg}")

    gwr_results = load_gwr_results(results_path)
    predict_from_gpkg(input_gpkg, output_gpkg, gwr_results)

if __name__ == '__main__':
    main()

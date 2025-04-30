#!/usr/bin/env python
import sys
import os
import logging

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd

# ——— Setup logging ———
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ——— Path to your pickled GWRResults ———
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'AI', 'gwr_results.pkl')

def load_gwr_results(path):
    """Load a pickled mgwr.gwr.GWRResults object."""
    try:
        results = joblib.load(path)
        logger.debug(f"Loaded GWRResults from {path}")
        return results
    except Exception as e:
        logger.error(f"Error loading GWR results: {e}")
        sys.exit(1)

def predict_from_gpkg(input_gpkg, output_gpkg, gwr_results):
    """
    Read input GeoPackage, attach gwr_results.predy as 'GWR_Prediction',
    and write out a new GeoPackage.
    """
    # 1) Load data
    try:
        gdf = gpd.read_file(input_gpkg)
        logger.debug(f"Loaded {len(gdf)} features from {input_gpkg}")
    except Exception as e:
        logger.error(f"Error reading '{input_gpkg}': {e}")
        sys.exit(1)

    # 2) Ensure an 'id' column (needed for Dash mapping)
    if 'id' not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf['id'] = gdf.index.astype(str)

    # 3) Grab the precomputed predictions
    if hasattr(gwr_results, 'predy'):
        preds = gwr_results.predy.flatten().tolist()
    else:
        logger.error("GWRResults has no .predy — cannot predict")
        sys.exit(1)

    # 4) Align lengths: pad with None if needed, or truncate
    n_feat = len(gdf)
    n_pred = len(preds)
    if n_pred < n_feat:
        logger.warning(f"Only {n_pred} predictions for {n_feat} features, padding with None")
        preds.extend([None] * (n_feat - n_pred))
    elif n_pred > n_feat:
        logger.warning(f"{n_pred} predictions for {n_feat} features, truncating")
        preds = preds[:n_feat]

    # 5) Attach to GeoDataFrame
    #    convert NaN floats → None so that Dash renders them blank
    clean = [None if (p is None or (isinstance(p, float) and np.isnan(p))) else p
             for p in preds]
    gdf['GWR_Prediction'] = clean

    # 6) Write out
    try:
        gdf.to_file(output_gpkg, driver='GPKG')
        logger.debug(f"Wrote GWR predictions to {output_gpkg}")
    except Exception as e:
        logger.error(f"Error writing '{output_gpkg}': {e}")
        sys.exit(1)

def main():
    # CLI args:  [1]=input_gpkg,  [2]=output_gpkg (optional)
    input_gpkg  = sys.argv[1] if len(sys.argv) > 1 else './AI/Rename_DataSet2.25.gpkg'
    if len(sys.argv) > 2:
        output_gpkg = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_gpkg)
        output_gpkg = f"{base}_with_gwr_predictions{ext}"

    logger.debug(f"Using GWR results: {RESULTS_PATH}")
    logger.debug(f"Input GeoPackage: {input_gpkg}")
    logger.debug(f"Output GeoPackage: {output_gpkg}")

    gwr_results = load_gwr_results(RESULTS_PATH)
    predict_from_gpkg(input_gpkg, output_gpkg, gwr_results)

if __name__ == '__main__':
    main()

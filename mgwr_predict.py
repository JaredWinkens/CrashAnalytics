#!/usr/bin/env python3
"""
predict_mgwr.py

A utility to load MGWR models and generate predictions into a GeoPackage,
using 'GEOIDFQ' as the tract identifier.
"""
import sys
import os
import logging
import pickle
import numpy as np
import geopandas as gpd

# ——— Setup logging ———
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ——— Paths ———
BASE_DIR = os.path.dirname(__file__)
MODEL_PKL = os.path.join(BASE_DIR, 'MGWR', 'mgwr_models.pkl')

def load_models(path):
    """Load the dict of MGWRResults objects."""
    try:
        with open(path, 'rb') as f:
            models = pickle.load(f)
        logger.debug(f"Loaded MGWR models from {path}")
        return models
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        sys.exit(1)

def predict(input_gpkg, output_gpkg, models):
    """Read input GPKG, generate MGWR_Prediction, and write out new GPKG."""
    try:
        gdf = gpd.read_file(input_gpkg)
        logger.debug(f"Loaded {len(gdf)} features from {input_gpkg}")
    except Exception as e:
        logger.error(f"Error reading '{input_gpkg}': {e}")
        sys.exit(1)

    # ───── Use GEOIDFQ as the primary ID ─────
    if 'GEOIDFQ' in gdf.columns:
        gdf = gdf.copy()
        gdf['id'] = gdf['GEOIDFQ'].astype(str)
        logger.debug("Using 'GEOIDFQ' column for id")
    elif 'id' not in gdf.columns:
        gdf = gdf.copy()
        gdf['id'] = gdf.index.astype(str)
        logger.debug("No 'GEOIDFQ' or 'id' column found; using index as id")

    # prepare an array of NaNs for predictions
    preds = np.full(len(gdf), np.nan, dtype=float)

    # ───── Loop over each region and assign predictions ─────
    for region, result in models.items():
        mask = (gdf['Region'] == region)
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            logger.warning(f"No features for region '{region}'")
            continue

        region_preds = result.predy.flatten()
        if len(region_preds) != idx.size:
            logger.warning(
                f"Region '{region}': {len(region_preds)} preds but {idx.size} features"
            )
            n = min(len(region_preds), idx.size)
            preds[idx[:n]] = region_preds[:n]
        else:
            preds[idx] = region_preds

    # attach predictions, converting NaN→None
    gdf['MGWR_Prediction'] = [None if np.isnan(x) else float(x) for x in preds]

    try:
        gdf.to_file(output_gpkg, driver='GPKG')
        logger.debug(f"Wrote MGWR predictions to {output_gpkg}")
    except Exception as e:
        logger.error(f"Error writing '{output_gpkg}': {e}")
        sys.exit(1)

def main():
    # CLI args: [1]=input, [2]=output (optional)
    input_gpkg = sys.argv[1] if len(sys.argv) > 1 else './MGWR/merged.gpkg'
    if len(sys.argv) > 2:
        output_gpkg = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_gpkg)
        output_gpkg = f"{base}_with_mgwr_predictions{ext}"

    logger.debug(f"Input: {input_gpkg}")
    logger.debug(f"Output: {output_gpkg}")

    models = load_models(MODEL_PKL)
    predict(input_gpkg, output_gpkg, models)

if __name__ == '__main__':
    main()

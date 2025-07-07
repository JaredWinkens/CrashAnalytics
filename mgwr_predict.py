#!/usr/bin/env python3
"""
predict_mgwr.py

A utility to load MGWR models and generate predictions into a GeoPackage,
using 'GEOIDFQ' as the tract identifier. Recomputes MGWR_Prediction from updated inputs and writes directly without merging.
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
    # 1. Read the updated input GeoPackage
    try:
        gdf = gpd.read_file(input_gpkg)
        logger.debug(f"Loaded {len(gdf)} features from {input_gpkg}")
    except Exception as e:
        logger.error(f"Error reading '{input_gpkg}': {e}")
        sys.exit(1)

    # 2. Log region distribution
    try:
        region_counts = gdf['Region'].value_counts()
        logger.debug("Regions and counts:\n%s", region_counts)
    except Exception:
        logger.debug("Could not compute region counts.")

    # 3. Static mapping of region-specific features using actual column names
    REGION_FIELDS = {
        "A": [
            "UNEMPPCT", "pct_residential", "pct_industrial", "pct_retail",
            "pct_commercial", "AADT", "BikingTrips.Start.", "BikingTrips.End.",
            "CarpoolTrips.Start.", "PublicTransitTrips.Start.", "PublicTransitTrips.End.",
            "AvgCommuteMiles.Start."
        ],
        "B": [
            "PEOPCOLORPCT", "UNEMPPCT", "pct_residential", "pct_industrial",
            "pct_retail", "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
            "BikingTrips.Start.", "WalkingTrips.End.", "PublicTransitTrips.Start.",
            "AvgCommuteMiles.End."
        ],
        "C": [
            "UNEMPPCT", "pct_residential", "pct_industrial", "pct_retail",
            "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
            "BikingTrips.Start.", "WalkingTrips.End.", "PublicTransitTrips.Start.",
            "PublicTransitTrips.End.", "AvgCommuteMiles.Start."
        ],
        "D": [
            "PEOPCOLORPCT", "pct_residential", "pct_industrial", "pct_retail",
            "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
            "BikingWalkingMiles.End.", "BikingTrips.Start.", "BikingTrips.End.",
            "CarpoolTrips.End.", "PublicTransitTrips.End.", "AvgCommuteMiles.Start."
        ],
        "E": [
            "PEOPCOLORPCT", "UNEMPPCT", "pct_residential", "pct_industrial",
            "pct_retail", "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
            "BikingWalkingMiles.End.", "BikingTrips.Start.", "WalkingTrips.Start.",
            "PublicTransitTrips.Start.", "AvgCommuteMiles.End."
        ],
        "FG": [
            "UNEMPPCT", "pct_residential", "pct_industrial", "pct_retail",
            "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
            "BikingWalkingMiles.End.", "BikingTrips.Start.", "BikingTrips.End.",
            "CarpoolTrips.End.", "PublicTransitTrips.Start.", "PublicTransitTrips.End.",
            "AvgCommuteMiles.End."
        ],
        "HIJ": [
            "UNEMPPCT", "DISABILITYPCT", "pct_residential", "pct_industrial",
            "pct_retail", "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
            "BikingTrips.Start.", "BikingTrips.End.", "CarpoolTrips.Start.",
            "CarpoolTrips.End.", "PublicTransitTrips.Start.", "PublicTransitTrips.End."
        ]
    }

    # 4. Prepare predictions array
    preds = np.full(len(gdf), np.nan, dtype=float)

    # 5. Loop through regions and compute predictions
    for region, result in models.items():
        features = REGION_FIELDS.get(region)
        if not features:
            logger.warning(f"No feature list for region '{region}'")
            continue

        mask = gdf['Region'] == region
        if not mask.any():
            continue

        # Ensure feature columns exist, filling missing with NaN
        for f in features:
            if f not in gdf.columns:
                logger.warning(f"Missing feature '{f}', filling with NaN")
                gdf[f] = np.nan

        X = gdf.loc[mask, features].to_numpy(dtype=float)
        coefs = result.params

        # Add intercept column if required
        if coefs.ndim == 2 and coefs.shape[1] == X.shape[1] + 1:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        elif coefs.ndim != 2 or coefs.shape[1] not in (X.shape[1], X.shape[1] + 1):
            logger.warning(f"Coef shape {coefs.shape} incompatible with X shape {X.shape}")
            continue

        # Align coefficient rows
        if coefs.shape[0] != X.shape[0]:
            coefs = coefs[: X.shape[0], :]

        # Compute predictions
        preds_region = np.einsum('ij,ij->i', coefs, X)
        preds[mask] = preds_region

    # 6. Clamp to zero and assign
    preds = np.clip(preds, 0, None)
    gdf['MGWR_Prediction'] = [None if np.isnan(x) else float(x) for x in preds]

    # 7. Ensure an 'id' column for Dash
    if 'GEOIDFQ' in gdf.columns:
        gdf['id'] = gdf['GEOIDFQ'].astype(str)
    else:
        gdf['id'] = gdf.index.astype(str)

    # 8. Write the updated GeoPackage directly
    try:
        gdf.to_file(output_gpkg, driver='GPKG')
        logger.debug(f"Wrote updated GPKG to {output_gpkg}")
    except Exception as e:
        logger.error(f"Error writing '{output_gpkg}': {e}")
        sys.exit(1)


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else './MGWR/merged.gpkg'
    out = sys.argv[2] if len(sys.argv) > 2 else inp.replace('.gpkg', '_with_all_tracts.gpkg')
    logger.debug(f"Input: {inp}, Output: {out}")
    models = load_models(MODEL_PKL)
    predict(inp, out, models)

if __name__ == '__main__':
    main()

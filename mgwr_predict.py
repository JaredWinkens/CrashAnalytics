#!/usr/bin/env python3
"""
predict_mgwr.py

A utility to load MGWR models and generate predictions into a GeoPackage,
using 'GEOIDFQ' as the tract identifier, then stitch those predictions onto the full set of tracts so that
missing tracts remain in the output with NaN for MGWR_Prediction, carry over all original variables, and include an 'id' column and 'Region' for app use.
Handles duplicate suffixes by preferring imported values (_y) over geometry defaults (_x).
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
FULL_TRACTS_GPKG = os.path.join(BASE_DIR, 'data', 'TractData.gpkg')


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
    # 1. Read prediction input
    try:
        gdf_in = gpd.read_file(input_gpkg)
        logger.debug(f"Loaded {len(gdf_in)} features from {input_gpkg}")
    except Exception as e:
        logger.error(f"Error reading '{input_gpkg}': {e}")
        sys.exit(1)

    gdf_in = gdf_in.copy()
    # ensure id
    gdf_in['id'] = gdf_in.get('GEOIDFQ', gdf_in.index.astype(str)).astype(str)

    # build predictions
    preds = np.full(len(gdf_in), np.nan, dtype=float)
    for region, result in models.items():
        mask = gdf_in.get('Region') == region
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            logger.warning(f"No features for region '{region}'")
            continue
        region_preds = result.predy.flatten()
        n = min(len(region_preds), idx.size)
        preds[idx[:n]] = region_preds[:n]
    preds = np.clip(preds, 0, None)
    gdf_in['MGWR_Prediction'] = [None if np.isnan(x) else float(x) for x in preds]

    # 2. Read full tracts
    try:
        gdf_all = gpd.read_file(FULL_TRACTS_GPKG)
        logger.debug(f"Loaded {len(gdf_all)} full tracts from {FULL_TRACTS_GPKG}")
    except Exception as e:
        logger.error(f"Error reading full tract layer: {e}")
        sys.exit(1)

    # 3. Merge on GEOIDFQ
    cols_pred = [c for c in gdf_in.columns if c != 'geometry']
    gdf_pred = gdf_in[cols_pred]
    gdf_full = gdf_all.merge(
        gdf_pred,
        on='GEOIDFQ', how='left',
        suffixes=('_x','_y')
    )
    # ensure id
    gdf_full['id'] = gdf_full['GEOIDFQ'].astype(str)

    # 4. Resolve duplicates: drop *_x, rename *_y to base name
    drop_cols = [c for c in gdf_full.columns if c.endswith('_x')]
    rename_map = {c: c[:-2] for c in gdf_full.columns if c.endswith('_y')}
    if drop_cols:
        logger.debug(f"Dropping columns: {drop_cols}")
        gdf_full = gdf_full.drop(columns=drop_cols)
    if rename_map:
        logger.debug(f"Renaming columns: {rename_map}")
        gdf_full = gdf_full.rename(columns=rename_map)
        # 4.5. For any tracts still missing a Region, assign default = 'B'
    if 'Region' in gdf_full.columns:
        gdf_full['Region'] = gdf_full['Region'].fillna('B')
    else:
        gdf_full['Region'] = 'B'

    # 5. Write output
    try:
        gdf_full.to_file(output_gpkg, driver='GPKG')
        logger.debug(f"Wrote merged GPKG to {output_gpkg}")
    except Exception as e:
        logger.error(f"Error writing '{output_gpkg}': {e}")
        sys.exit(1)


def main():
    inp = sys.argv[1] if len(sys.argv)>1 else './MGWR/merged.gpkg'
    out = sys.argv[2] if len(sys.argv)>2 else inp.replace('.gpkg','_with_all_tracts.gpkg')
    logger.debug(f"Input: {inp}, Output: {out}")
    models = load_models(MODEL_PKL)
    predict(inp, out, models)

if __name__=='__main__':
    main()

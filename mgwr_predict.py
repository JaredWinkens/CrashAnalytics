import sys
import os
import logging
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# ——— Setup logging ———
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ——— Paths ———
BASE_DIR = os.path.dirname(__file__)
MODEL_PKL = os.path.join(BASE_DIR, 'MGWR', 'mgwr_models.pkl')


def load_models(path):
    # Load the dict of MGWRResults objects.
    try:
        with open(path, 'rb') as f:
            #models is expected to be a dict mapping region code
            models = pickle.load(f)
        logger.debug(f"Loaded MGWR models from {path}")
        return models
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        sys.exit(1)


def predict(input_gpkg, output_gpkg, models):
    # 1. Read the updated input GeoPackage from the user
    try:
        gdf = gpd.read_file(input_gpkg)
        logger.debug(f"Loaded {len(gdf)} features from {input_gpkg}")
    except Exception as e:
        logger.error(f"Error reading '{input_gpkg}': {e}")
        sys.exit(1)

    # 2. Check for existing predictions and only process tracts that already have valid predictions
    if 'MGWR_Prediction' in gdf.columns:
        # Identify tracts that already have valid (non-NaN) predictions
        has_existing_prediction = gdf['MGWR_Prediction'].notna()
        tracts_to_process = has_existing_prediction.sum()
        tracts_to_skip = (~has_existing_prediction).sum()
        
        logger.debug(f"Found existing MGWR_Prediction column:")
        logger.debug(f"  Tracts with valid predictions: {tracts_to_process}")
        logger.debug(f"  Tracts with NaN predictions (will skip): {tracts_to_skip}")
        
        if tracts_to_process == 0:
            logger.warning("No tracts have existing predictions - nothing to process!")
            # Just ensure id column and save
            if 'GEOIDFQ' in gdf.columns:
                gdf['id'] = gdf['GEOIDFQ'].astype(str)
            else:
                gdf['id'] = gdf.index.astype(str)
            gdf.to_file(output_gpkg, driver='GPKG')
            return
    else:
        # No existing prediction column - process all tracts (original behavior)
        logger.debug("No existing MGWR_Prediction column found - will process all tracts")
        has_existing_prediction = pd.Series(True, index=gdf.index)

    # 3. Log region distribution (mainly a testing feature)
    try:
        region_counts = gdf['Region'].value_counts()
        logger.debug("Regions and counts:\n%s", region_counts)
    except Exception:
        logger.debug("Could not compute region counts.")

    # 4. Static mapping of region-specific features, each region uses different variables
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

    # 5. Initialize predictions array - preserve existing predictions
    if 'MGWR_Prediction' in gdf.columns:
        preds = gdf['MGWR_Prediction'].to_numpy(dtype=float)
    else:
        preds = np.full(len(gdf), np.nan, dtype=float)

    # 6. Loop through regions and compute predictions ONLY for tracts that already have valid predictions
    total_processed = 0
    for region, result in models.items():
        features = REGION_FIELDS.get(region)
        if not features:
            logger.warning(f"No feature list for region '{region}'")
            continue
        
        # mask selects only rows in gdf matching current region AND have existing predictions
        region_mask = gdf['Region'] == region
        processable_mask = region_mask & has_existing_prediction
        
        if not processable_mask.any():
            logger.debug(f"No processable tracts for region '{region}' (no existing predictions)")
            continue

        region_count = region_mask.sum()
        processable_count = processable_mask.sum()
        logger.debug(f"Region '{region}': {processable_count}/{region_count} tracts have existing predictions")

        # Ensure feature columns exist
        for f in features:
            if f not in gdf.columns:
                logger.warning(f"Missing feature '{f}', filling with NaN")
                gdf[f] = np.nan

        X = gdf.loc[processable_mask, features].to_numpy(dtype=float)
        #holds the MGWR coefficient surface
        coefs = result.params

        # Handle intercept if needed
        if coefs.ndim == 2 and coefs.shape[1] == X.shape[1] + 1:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        elif coefs.ndim != 2 or coefs.shape[1] not in (X.shape[1], X.shape[1] + 1):
            logger.warning(f"Coef shape {coefs.shape} incompatible with X shape {X.shape}")
            continue

        # align coefficients by spatial matching 
        try:
            orig_coords = getattr(result, 'coords', None)
            if orig_coords is None:
                orig_coords = result.model.coords
        except Exception as e:
            logger.error(f"Could not retrieve original MGWR coords: {e}")
            sys.exit(1)

        subset = gdf.loc[processable_mask]
        #for each tract calc its geometric centroid
        centroids = subset.geometry.centroid
        #put these in array of shapes to use in nearest neighbor search
        subset_coords = np.vstack([[pt.x, pt.y] for pt in centroids])
        

        # 1) compute each tract's raw local prediction nearest‐neighbor
        tree = cKDTree(orig_coords)
        _, idxs_nn = tree.query(subset_coords, k=1)
        coefs_local = result.params[idxs_nn, :]
        raw_preds = np.einsum('ij,ij->i', coefs_local, X)


        # 2) smooth predictions across ALL OTHER tracts in this region
        tree_reg = cKDTree(subset_coords)
        n = subset_coords.shape[0]
        
        if n > 1:
            # get distances & indices for all tracts (first index always self)
            dists_all, idxs_all = tree_reg.query(subset_coords, k=n)

            # drop the self at [:,0], keep neighbors only
            dists = dists_all[:, 1:]   # shape (n, n-1)
            idxs  = idxs_all[:, 1:]    # shape (n, n-1)

            eps = 1e-6  # to avoid division by zero if two centroids coincide
            weights = 1.0 / (dists + eps)
            # normalize so each row sums to 1
            weights /= weights.sum(axis=1)[:, None]

            # weighted average of *neighbors'* raw_preds
            smoothed_preds = (raw_preds[idxs] * weights).sum(axis=1)

            # blend them 50/50
            alpha = 0.5   # 0 no smoothing, 1 full smoothing
            final_preds = (1 - alpha) * raw_preds + alpha * smoothed_preds
        else:
            # Only one tract in region - no smoothing possible
            final_preds = raw_preds

        # assign back to the specific indices
        preds[processable_mask] = final_preds
        total_processed += len(final_preds)

    logger.debug(f"Total tracts processed: {total_processed}")

    # 7. Clamp to zero and assign (dont let values go negative)
    preds = np.clip(preds, 0, None)
    gdf['MGWR_Prediction'] = [None if np.isnan(x) else float(x) for x in preds]

    # 8. Ensure an 'id' column for Dash (old and will be phased out when i get a chance to fix it)
    if 'GEOIDFQ' in gdf.columns:
        gdf['id'] = gdf['GEOIDFQ'].astype(str)
    else:
        gdf['id'] = gdf.index.astype(str)

    # 9. Write the updated GeoPackage
    try:
        gdf.to_file(output_gpkg, driver='GPKG')
        logger.debug(f"Wrote updated GPKG to {output_gpkg}")
        
        # Final statistics
        final_valid = np.sum(~np.isnan(preds))
        final_nan = np.sum(np.isnan(preds))
        logger.debug(f"Final predictions: {final_valid} valid, {final_nan} NaN")
        
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
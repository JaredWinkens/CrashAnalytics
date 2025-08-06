import sys, os, pickle, logging
import numpy as np
import pandas as pd
import geopandas as gpd
from mgwr.gwr    import MGWR
from mgwr.sel_bw import Sel_BW

# ——— Logging ———
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('mgwr').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ——— Paths ———
BASE_DIR    = os.path.dirname(__file__)
MODEL_PKL   = os.path.join(BASE_DIR, 'MGWR', 'mgwr_models.pkl')
MERGED_GPKG = os.path.join(BASE_DIR, 'MGWR', 'merged.gpkg')

# ——— Region→Features mapping ———
REGION_FIELDS = {
    "A": [ "UNEMPPCT", "pct_residential", "pct_industrial", "pct_retail",
           "pct_commercial", "AADT", "BikingTrips.Start.", "BikingTrips.End.",
           "CarpoolTrips.Start.", "PublicTransitTrips.Start.",
           "PublicTransitTrips.End.", "AvgCommuteMiles.Start." ],
    "B": [ "PEOPCOLORPCT", "UNEMPPCT", "pct_residential", "pct_industrial",
           "pct_retail", "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
           "BikingTrips.Start.", "WalkingTrips.End.",
           "PublicTransitTrips.Start.", "AvgCommuteMiles.End." ],
    "C": [ "UNEMPPCT", "pct_residential", "pct_industrial", "pct_retail",
           "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
           "BikingTrips.Start.", "WalkingTrips.End.",
           "PublicTransitTrips.Start.", "PublicTransitTrips.End.",
           "AvgCommuteMiles.Start." ],
    "D": [ "PEOPCOLORPCT", "pct_residential", "pct_industrial", "pct_retail",
           "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
           "BikingWalkingMiles.End.", "BikingTrips.Start.",
           "BikingTrips.End.", "CarpoolTrips.End.", "PublicTransitTrips.End.",
           "AvgCommuteMiles.Start." ],
    "E": [ "PEOPCOLORPCT", "UNEMPPCT", "pct_residential", "pct_industrial",
           "pct_retail", "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
           "BikingWalkingMiles.End.", "BikingTrips.Start.", "WalkingTrips.Start.",
           "PublicTransitTrips.Start.", "AvgCommuteMiles.End." ],
    "FG":[ "UNEMPPCT", "pct_residential", "pct_industrial", "pct_retail",
           "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
           "BikingWalkingMiles.End.", "BikingTrips.Start.", "BikingTrips.End.",
           "CarpoolTrips.End.", "PublicTransitTrips.Start.",
           "PublicTransitTrips.End.", "AvgCommuteMiles.End." ],
    "HIJ":[ "UNEMPPCT", "DISABILITYPCT", "pct_residential", "pct_industrial",
           "pct_retail", "pct_commercial", "AADT", "BikingWalkingMiles.Start.",
           "BikingTrips.Start.", "BikingTrips.End.", "CarpoolTrips.Start.",
           "CarpoolTrips.End.", "PublicTransitTrips.Start.",
           "PublicTransitTrips.End." ]
}

def predict(county_gpkg: str, out_gpkg: str):
    # 1) Load the county GPKG, detect its region code
    county = gpd.read_file(county_gpkg)
    if county.empty:
        raise RuntimeError(f"{county_gpkg} has no features")
    
    # Store the original CRS from county data
    county_crs = county.crs
    logger.info(f"County CRS: {county_crs}")
    
    # Log the county data
    logger.info(f"Loaded county GPKG with {len(county)} tracts")
    logger.info(f"Columns in county GPKG: {list(county.columns)}")
    
    # Check for NaN predictions
    if 'MGWR_Prediction' in county.columns:
        nan_count = county['MGWR_Prediction'].isna().sum()
        logger.info(f"County has {nan_count} tracts with NaN predictions")
    
    region_code = county.loc[0, "Region"]
    logger.info("County file region = %s", region_code)

    # 2) Load full merged.gpkg, store original CRS, then project
    full = gpd.read_file(MERGED_GPKG)
    full_crs = full.crs
    logger.info(f"Merged GPKG CRS: {full_crs}")
    
    # Project to 3857 for calculations
    full = full.to_crs(epsg=3857)
    
    # First, let's see if our county tracts exist in the full dataset
    county_geoids = set(county['GEOIDFQ'].values)
    full_geoids = set(full['GEOIDFQ'].values)
    missing_in_full = county_geoids - full_geoids
    if missing_in_full:
        logger.warning(f"These county tracts are NOT in merged.gpkg: {missing_in_full}")
    
    mask = full["Region"] == region_code
    region_gdf = full[mask].reset_index(drop=True)
    n = len(region_gdf)
    logger.info("→ %d tracts in region %s", n, region_code)
    
    # Collect new tracts to add
    new_tracts = []
    
    # Now we need to add our edited county data to the region data
    logger.info("Updating region data with edited county values...")
    
    # For each tract in county that was edited
    for idx, county_tract in county.iterrows():
        geoid = county_tract['GEOIDFQ']
        
        # Find this tract in region_gdf
        region_idx = region_gdf[region_gdf['GEOIDFQ'] == geoid].index
        
        if len(region_idx) == 0:
            logger.warning(f"Tract {geoid} not found in region data - will add it")
            # Create a copy of the tract and set its CRS to match the original full data
            new_tract = county_tract.copy()
            new_tracts.append(new_tract)
        else:
            # Update the values in region_gdf with edited values from county
            logger.info(f"Updating tract {geoid} with edited values")
            for col in REGION_FIELDS[region_code]:
                if col in county.columns:
                    region_gdf.loc[region_idx[0], col] = county_tract[col]
    
    # Add new tracts if any
    if new_tracts:
        logger.info(f"Adding {len(new_tracts)} new tracts to region data")
        # Create a GeoDataFrame from new tracts with the original county CRS
        new_gdf = gpd.GeoDataFrame(new_tracts, crs=county_crs)
        
        # First project new tracts to match the original full CRS, then to 3857
        if county_crs != full_crs:
            new_gdf = new_gdf.to_crs(full_crs)
        new_gdf = new_gdf.to_crs(epsg=3857)
        
        # Concatenate
        region_gdf = pd.concat([region_gdf, new_gdf], ignore_index=True)
    
    n = len(region_gdf)
    logger.info(f"After updates: {n} tracts in region dataset")

    # 3) Build coords, y, X for that region
    coords = np.vstack([[pt.x, pt.y] for pt in region_gdf.geometry.centroid])
    y      = region_gdf["VRU_rate"].to_numpy(dtype=float).reshape(-1,1)
    
    # Log y values
    logger.info(f"Y (VRU_rate) stats: min={y.min()}, max={y.max()}, mean={y.mean()}, NaN count={np.isnan(y).sum()}")
    
    feats  = REGION_FIELDS[region_code]
    
    # Check if all required features exist
    missing_feats = [f for f in feats if f not in region_gdf.columns]
    if missing_feats:
        logger.error(f"Missing features in region data: {missing_feats}")
        logger.info(f"Available columns: {list(region_gdf.columns)}")
        raise RuntimeError(f"Missing required features: {missing_feats}")
    
    X = region_gdf[feats].to_numpy(dtype=float)
    
    # Log feature matrix stats
    logger.info(f"X shape: {X.shape}")
    logger.info(f"X has {np.isnan(X).sum()} NaN values")
    
    # intercept?
    with open(MODEL_PKL, "rb") as f:
        saved = pickle.load(f)[region_code]
    if saved.params.shape[1] == X.shape[1] + 1:
        X = np.hstack([np.ones((n,1)), X])
        logger.info("Added intercept column")

    # 4) Bandwidth selection
    import time
    start = time.time()
    selector = Sel_BW(
        coords, y, X,
        fixed=False,
        kernel="bisquare",
        multi=True,       # multi‐scale search
        constant=True,
        spherical=False
    )
    selector.search(
        criterion="AICc",
        search_method="golden_section",
        bw_min=10,    # adjust these bounds if you like
        bw_max=200
    )
    logger.info(
        "Recomputed bandwidth(s): %s (took %.1f s)",
        selector.bw, time.time() - start
    )

    # 5) Full MGWR back‐fit using the freshly‐found bandwidth(s)
    logger.info("Starting MGWR back-fitting on %d tracts…", n)
    gwr = MGWR(
        coords, y, X, selector,
        fixed     = saved.model.fixed,
        kernel    = saved.model.kernel,
        constant  = saved.model.constant,
        spherical = saved.model.spherical,
        hat_matrix= saved.model.hat_matrix
    )
    res = gwr.fit()

    # 6) Extract predictions for region, then filter to county only
    preds = np.clip(res.predy.flatten(), 0, None)
    region_gdf["MGWR_Prediction"] = preds
    
    logger.info(f"Generated {len(preds)} predictions")
    logger.info(f"Prediction stats: min={preds.min()}, max={preds.max()}, mean={preds.mean()}")

    # 7) Rebuild county_out so it has exactly one MGWR_Prediction column
    #    a) drop any old MGWR_Prediction columns
    county_out = county.drop(
        columns=[c for c in county.columns if c.startswith("MGWR_Prediction")],
        errors='ignore'
    )
    #    b) merge in the fresh predictions
    county_out = county_out.merge(
        region_gdf[["GEOIDFQ", "MGWR_Prediction"]],
        on="GEOIDFQ", how="left"
    )
    
    # Log merge results
    merged_count = county_out['MGWR_Prediction'].notna().sum()
    logger.info(f"After merge: {merged_count} tracts have predictions")
    
    if 'MGWR_Prediction' in county_out.columns:
        still_nan = county_out['MGWR_Prediction'].isna().sum()
        if still_nan > 0:
            logger.warning(f"Still have {still_nan} tracts with NaN predictions after retrain")
            nan_geoids = county_out[county_out['MGWR_Prediction'].isna()]['GEOIDFQ'].tolist()
            logger.warning(f"NaN GEOIDFQs: {nan_geoids}")

    # 8) Write out a new GPKG with exactly the same county schema + new preds
    county_out.to_file(out_gpkg, driver="GPKG")
    logger.info("Wrote retrained county GPKG: %s (%d records)", out_gpkg, len(county_out))


def main():
    if len(sys.argv) != 3:
        print("Usage: python mgwr_retrain_county.py <county_editable.gpkg> <out.gpkg>")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2])

if __name__=="__main__":
    main()
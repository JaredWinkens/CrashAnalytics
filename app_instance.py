import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import base64
from flask_caching import Cache
import os
import dash_auth
import logging
import pandas as pd

stylesheets = [
    dbc.themes.BOOTSTRAP,
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'
    ]

# Initialize the Dash app with Bootstrap for styling
app = dash.Dash(
    __name__,
    external_stylesheets=stylesheets,
    # Suppress callback exceptions to allow for callbacks on dynamically generated components
    # This is often needed when using layouts defined in separate files.
    suppress_callback_exceptions=True
)

app.title = 'Crash Data Analytics'
server = app.server

secret_key = base64.b64encode(os.urandom(30)).decode('utf-8')
auth = dash_auth.BasicAuth(
    app,
    {
        'winkenj': 'SunyPoly2025',
        'karimpa': 'TrafficSafety2025'
    },
    secret_key=secret_key,
)

# Initialize caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple'  
})

# Define cache timeout (e.g., 1 hour)
CACHE_TIMEOUT = 60 * 60

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ALL_FIELDS = [
    ("DEMOGIDX_5",     "Demographic Index (5-yr ACS)",        0.01),
    ("PEOPCOLORPCT",   "People of Color (%)",                0.01),
    ("UNEMPPCT",       "Unemployment Rate (%)",             0.01),
    ("pct_residential","Residential Land Use (%)",          0.01),
    ("pct_industrial", "Industrial Land Use (%)",           0.01),
    ("pct_retail",     "Retail Land Use (%)",               0.01),
    ("pct_commercial", "Commercial Land Use (%)",           0.01),
    ("AADT",           "Annual Average Daily Traffic",       0.01),
    ("Commute_TripMiles_TripStart_avg","Avg Commute Distance (Start, mi)",0.01),
    ("Commute_TripMiles_TripEnd_avg",  "Avg Commute Distance (End, mi)",  0.01),
    ("ACSTOTPOP",      "Total Population (ACS)",             1),
    ("DEMOGIDX_2",     "Demographic Index (2-yr ACS)",      0.01),
    ("PovertyPop",     "Poverty Population",                 1),
    ("DISABILITYPCT",  "Disability (%)",                     0.01),
    ("BikingTrips(Start)","Biking Trips (Start)",           1),
    ("BikingTrips(End)",  "Biking Trips (End)",             1),
    ("CarpoolTrips(Start)","Carpool Trips (Start)",         1),
    ("CarpoolTrips(End)",  "Carpool Trips (End)",           1),
    ("CommercialFreightTrips(Start)","Commercial Freight Trips (Start)",1),
    ("CommercialFreightTrips(End)",  "Commercial Freight Trips (End)", 1),
    ("WalkingTrips(Start)",          "Walking Trips (Start)",        1),
    ("WalkingTrips(End)",            "Walking Trips (End)",          1),
    ("PublicTransitTrips(Start)",    "Public Transit Trips (Start)",1),
    ("PublicTransitTrips(End)",      "Public Transit Trips (End)",   1),
    ("AvgCommuteMiles(Start)",       "Avg Commute Miles (Start)",    0.01),
    ("AvgCommuteMiles(End)",         "Avg Commute Miles (End)",      0.01),
    ("BikingWalkingMiles(Start)",    "Biking & Walking Miles (Start)",0.01),
    ("BikingWalkingMiles(End)",      "Biking & Walking Miles (End)", 0.01),
]

#quick lookup dicts
LABELS = { var: label for var, label, step in ALL_FIELDS }
STEPS  = { var: step  for var, label, step in ALL_FIELDS }

REGION_FIELDS = {
    "A": [
        "UNEMPPCT",
        "pct_residential",
        "pct_industrial",
        "pct_retail",
        "pct_commercial",
        "AADT",
        "BikingTrips(Start)",
        "BikingTrips(End)",
        "CarpoolTrips(Start)",
        "PublicTransitTrips(Start)",
        "PublicTransitTrips(End)",
        "AvgCommuteMiles(Start)"
    ],
    "B": [
        "PEOPCOLORPCT",
        "UNEMPPCT",
        "pct_residential",
        "pct_industrial",
        "pct_retail",
        "pct_commercial",
        "AADT",
        "BikingWalkingMiles(Start)",
        "BikingTrips(Start)",
        "WalkingTrips(End)",
        "PublicTransitTrips(Start)",
        "AvgCommuteMiles(End)"
    ],
    "C": [
        "UNEMPPCT",
        "pct_residential",
        "pct_industrial",
        "pct_retail",
        "pct_commercial",
        "AADT",
        "BikingWalkingMiles(Start)",
        "BikingTrips(Start)",
        "WalkingTrips(End)",
        "PublicTransitTrips(End)",
        "AvgCommuteMiles(Start)"
    ],
    "D": [
        "PEOPCOLORPCT",
        "pct_residential",
        "pct_industrial",
        "pct_retail",
        "pct_commercial",
        "AADT",
        "BikingWalkingMiles(Start)",
        "BikingWalkingMiles(End)",
        "BikingTrips(Start)",
        "BikingTrips(End)",
        "CarpoolTrips(End)",
        "PublicTransitTrips(End)",
        "AvgCommuteMiles(Start)"
    ],
    "E": [
        "PEOPCOLORPCT",
        "UNEMPPCT",
        "pct_residential",
        "pct_industrial",
        "pct_retail",
        "pct_commercial",
        "AADT",
        "BikingWalkingMiles(Start)",
        "BikingWalkingMiles(End)",
        "BikingTrips(Start)",
        "WalkingTrips(Start)",
        "PublicTransitTrips(Start)",
        "AvgCommuteMiles(End)"
    ],
    "FG": [
        "UNEMPPCT",
        "pct_residential",
        "pct_industrial",
        "pct_retail",
        "pct_commercial",
        "AADT",
        "BikingWalkingMiles(Start)",
        "BikingWalkingMiles(End)",
        "BikingTrips(Start)",
        "BikingTrips(End)",
        "PublicTransitTrips(Start)",
        "PublicTransitTrips(End)",
        "AvgCommuteMiles(End)"
    ],
    "HIJ": [
        "UNEMPPCT",
        "DISABILITYPCT",
        "pct_residential",
        "pct_industrial",
        "pct_retail",
        "pct_commercial",
        "AADT",
        "BikingWalkingMiles(Start)",
        "BikingTrips(Start)",
        "BikingTrips(End)",
        "CarpoolTrips(Start)",
        "CarpoolTrips(End)",
        "PublicTransitTrips(Start)",
        "PublicTransitTrips(End)"
    ]
}

# helper to build a single field‐row
def make_field_row(var_id, label, step):
    return html.Div([
        html.Label(label, style={'fontSize':'12px','marginRight':'5px'}),
        dcc.Input(id=f"input_{var_id}", type="number", value=0, step=step, style={'width':'80px','fontSize':'12px'}),
        html.Button("+", id=f"plus_input_{var_id}", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
        html.Button("–", id=f"minus_input_{var_id}", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'})
    ], id=f"container_{var_id}", style={'marginBottom':'10px'}
)

DEFAULT_PRED_FILES = {
    'AI.py':  './AI/Large_DataSet2.25_with_predictions.gpkg',
    'AI2.py': './AI/Rename_DataSet2.25_with_gwr_predictions.gpkg',
    'mgwr_predict.py': './MGWR/merged_with_mgwr_predictions.gpkg'
}

# Global variable to store data by county
data_by_county = {}

# Define county coordinates globally for access in callbacks
county_coordinates = {}

@cache.memoize(timeout=CACHE_TIMEOUT)
def get_county_data(counties_selected):
    """
    Retrieve preloaded data for the specified counties.

    Parameters:
        counties_selected (list): The counties to retrieve data for.

    Returns:
        pd.DataFrame: Combined DataFrame for the selected counties.
    """
    if not isinstance(counties_selected, list):
        counties_selected = [counties_selected]

    if 'All' in counties_selected:
        # Return data for all counties
        combined_df = pd.concat(data_by_county.values(), ignore_index=True)
        logger.debug(f"'All' selected. Returning data for all counties with {len(combined_df)} records.")
        return combined_df

    # Else, return data for selected counties
    dfs = []
    for county in counties_selected:
        if county in data_by_county:
            dfs.append(data_by_county[county])
        else:
            logger.warning(f"No data found for county: {county}")
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

from utils import data_processing

data_folder = 'data'
data_final_file = os.path.join(data_folder, 'Data_Final_2024.csv')  # Data_Final.csv path
census_data_file = os.path.join(data_folder, 'TractData.gpkg')  # GeSoPackage file for Census Data

# ----------------------------
# 8.2. Load Data and Define Counties
# ----------------------------
data_final_df = data_processing.load_data_final(data_final_file)
unique_counties = data_final_df['County'].dropna().unique().tolist()
counties = unique_counties  # Dynamically set counties based on Data_Final.csv

logger.debug(f"Unique counties extracted from Data_Final.csv: {counties}")

unique_weather = sorted(data_final_df['WeatherCon'].dropna().unique().tolist())
unique_light = sorted(data_final_df['LightCon'].dropna().unique().tolist())
unique_road = sorted(data_final_df['RoadSurfac'].dropna().unique().tolist())
unique_crash_types = sorted(data_final_df['Crash_Type'].dropna().unique().tolist())

logger.debug(f"Unique Weather Conditions: {unique_weather}")
logger.debug(f"Unique Light Conditions: {unique_light}")
logger.debug(f"Unique Road Surface Conditions: {unique_road}")

# Define county coordinates (must include all counties)
county_coordinates = {
    'Albany': {'lat': 42.6526, 'lon': -73.7562},
    'Allegany': {'lat': 42.0411, 'lon': -78.1564},
    'Bronx': {'lat': 40.8448, 'lon': -73.8648},
    'Broome': {'lat': 42.1794, 'lon': -75.9085},
    'Cattaraugus': {'lat': 42.4659, 'lon': -78.3200},
    'Cayuga': {'lat': 42.7304, 'lon': -76.5494},
    'Chautauqua': {'lat': 42.2315, 'lon': -79.4690},
    'Chemung': {'lat': 42.1325, 'lon': -76.7713},
    'Chenango': {'lat': 42.2731, 'lon': -75.9852},
    'Clinton': {'lat': 44.7300, 'lon': -73.8292},
    'Columbia': {'lat': 42.1488, 'lon': -73.9336},
    'Cortland': {'lat': 42.6120, 'lon': -76.1834},
    'Delaware': {'lat': 42.3938, 'lon': -75.4204},
    'Dutchess': {'lat': 41.7548, 'lon': -73.5673},
    'Erie': {'lat': 42.8802, 'lon': -78.8784},
    'Essex': {'lat': 44.0067, 'lon': -73.8781},
    'Franklin': {'lat': 44.4118, 'lon': -74.1764},
    'Fulton': {'lat': 43.0475, 'lon': -74.2159},
    'Genesee': {'lat': 42.9833, 'lon': -77.5667},
    'Greene': {'lat': 42.3162, 'lon': -74.0525},
    'Hamilton': {'lat': 43.2271, 'lon': -74.6947},
    'Herkimer': {'lat': 43.0398, 'lon': -74.9036},
    'Jefferson': {'lat': 43.7793, 'lon': -75.6410},
    'Kings': {'lat': 40.6782, 'lon': -73.9442},
    'Lewis': {'lat': 43.0056, 'lon': -75.8972},
    'Livingston': {'lat': 42.7074, 'lon': -77.8454},
    'Madison': {'lat': 43.0703, 'lon': -75.3974},
    'Monroe': {'lat': 43.2150, 'lon': -77.6150},
    'Montgomery': {'lat': 42.6526, 'lon': -73.7562},
    'Nassau': {'lat': 40.7003, 'lon': -73.6544},
    'New York': {'lat': 40.7128, 'lon': -74.0060},
    'Niagara': {'lat': 43.3110, 'lon': -78.6764},
    'Oneida': {'lat': 43.1167, 'lon': -75.3833},
    'Onondaga': {'lat': 43.0481, 'lon': -76.1474},
    'Ontario': {'lat': 43.4190, 'lon': -77.5460},
    'Orange': {'lat': 41.2578, 'lon': -74.0636},
    'Orleans': {'lat': 43.2797, 'lon': -78.4310},
    'Oswego': {'lat': 43.4691, 'lon': -76.5485},
    'Otsego': {'lat': 42.3967, 'lon': -75.5833},
    'Putnam': {'lat': 41.3917, 'lon': -73.9403},
    'Queens': {'lat': 40.7282, 'lon': -73.7949},
    'Rensselaer': {'lat': 42.7506, 'lon': -73.8180},
    'Richmond': {'lat': 40.5795, 'lon': -74.1502},
    'Rockland': {'lat': 41.2129, 'lon': -74.0776},
    'St. Lawrence': {'lat': 44.6175, 'lon': -75.4075},
    'Saratoga': {'lat': 43.1167, 'lon': -73.7833},
    'Schenectady': {'lat': 42.8142, 'lon': -73.9396},
    'Schoharie': {'lat': 42.8000, 'lon': -74.2333},
    'Schuyler': {'lat': 42.3000, 'lon': -76.5000},
    'Seneca': {'lat': 42.6910, 'lon': -76.9161},
    'Steuben': {'lat': 42.0700, 'lon': -77.0650},
    'Suffolk': {'lat': 40.9776, 'lon': -72.6277},
    'Sullivan': {'lat': 41.6414, 'lon': -74.4970},
    'Tioga': {'lat': 42.1139, 'lon': -76.3317},
    'Tompkins': {'lat': 42.4577, 'lon': -76.5488},
    'Ulster': {'lat': 41.8995, 'lon': -74.0758},
    'Warren': {'lat': 41.7585, 'lon': -73.7272},
    'Washington': {'lat': 41.2737, 'lon': -73.8087},
    'Wayne': {'lat': 41.4016, 'lon': -74.2081},
    'Westchester': {'lat': 41.1496, 'lon': -73.7824},
    'Wyoming': {'lat': 41.3312, 'lon': -74.0104},
    'Yates': {'lat': 42.4207, 'lon': -76.5758},
}

missing_coords = set(counties) - set(county_coordinates.keys())
if missing_coords:
    logger.warning(f"Missing coordinates for counties: {missing_coords}")
    for county in missing_coords:
        county_coordinates[county] = {'lat': 40.7128, 'lon': -74.0060}

data_by_county = data_processing.load_all_data_optimized(
    data_final_file, counties
)

# Load Census_Tract_data from the GeoPackage file
census_polygons_by_county = data_processing.load_census_data(census_data_file)
logger.debug(f"Census polygons loaded for {len(census_polygons_by_county)} counties.")

available_road_classes = ['Urban:Major Collector', 'Urban:Principal Arterial - Interstate', 
    'Urban:Principal Arterial - Freeways & Expressways', 'Urban:Minor Arterial', 
    'Urban:Local', 'Urban:Principal Arterial - Other', 'LU:Principal Arterial - Interstate', '', 
    'LU:Principal Arterial - Freeways & Expressways', 'Rural:Minor Arterial', 'Rural:Major Collector', 
    'Rural:Local', 'Rural:Minor Collector', 'Rural:Principal Arterial - Freeways & Expressways', 
    'Urban:Minor Collector', 'Rural:Principal Arterial - Other', 'Rural:Principal Arterial - Interstate', 
    'LU:Principal Arterial - Other', ':Principal Arterial - Other']
available_road_classes.sort()

available_intersection_classes = ['2-way', '3-way', '4-way', '5-way', '6-way', '7-way', '8-way']

globals()['county_coordinates'] = county_coordinates
globals()['census_polygons_by_county'] = census_polygons_by_county
globals()['data_by_county'] = data_by_county
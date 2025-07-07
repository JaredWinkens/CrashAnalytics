import base64
import dash
from dash import Dash, DiskcacheManager, dcc, html, Input, Output, State, callback_context, ctx, clientside_callback, MATCH, ALL
import datetime
import diskcache
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go  
import pandas as pd
import os
import sys
import logging
from flask_caching import Cache
from dash.exceptions import PreventUpdate
from shapely import wkt  
from shapely.geometry import mapping  
import geopandas as gpd 
import math
import json
import subprocess
import chatbot.chatbot_layout as chatbotlayout
#import chatbot.chatbot_v3 as chatbotv3
import analyzer.map_analyzer as map_analyzer
import concurrent.futures
import analyzer.streetview_analyzer as streetview
#from chatbot.mcp_client import get_gemini_response_from_mcp
import chatbot.chatbot_v4 as chatbotv4
import asyncio

def call_with_timeout(func, timeout, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Function '{func.__name__}' timed out after {timeout} seconds.")
            return f"Function '{func.__name__}' timed out after {timeout} seconds."
        except Exception as e:
            print(f"Function '{func.__name__}' raised an exception: {e}")
            return f"Function '{func.__name__}' raised an exception: {e}"
# disk_cache = diskcache.Cache("./cache")
# background_callback_manager = DiskcacheManager(disk_cache)       
# all editable fields for prediction tab
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

# region key for mgwr cause a better way eludes me
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

# ----------------------------
# 1. Setup Logging
# ----------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------
# 2. Data Loading and Preprocessing
# ----------------------------


def copy_county_gpkg(county, source_gpkg, dest_folder):
    """
    Extract just one county’s features into a new editable GPKG,
    filtering on CNTY_NAME or CountyName as available.
    """
    try:
        gdf = gpd.read_file(source_gpkg)

        # figure out which county column to use
        if 'CNTY_NAME' in gdf.columns:
            col = 'CNTY_NAME'
            gdf[col] = (
                gdf[col]
                   .str.replace(" County", "", regex=False)
                   .str.strip()
                   .str.title()
            )
        elif 'CountyName' in gdf.columns:
            col = 'CountyName'
            gdf[col] = (
                gdf[col]
                   .str.replace(" County", "", regex=False)
                   .str.strip()
                   .str.title()
            )
        else:
            logger.error(f"No CNTY_NAME or CountyName column in {source_gpkg}")
            raise PreventUpdate

        # filter to just that county
        county_gdf = gdf[gdf[col] == county]
        if county_gdf.empty:
            logger.error(f"No data found for county: {county} in {source_gpkg}")
            raise PreventUpdate

        # ensure an 'id' column for downstream callbacks
        county_gdf = county_gdf.copy()
        if 'id' not in county_gdf.columns:
            county_gdf['id'] = county_gdf.index.astype(str)

        # write out the editable file
        dest_file = os.path.join(dest_folder, f"{county}_editable.gpkg")
        county_gdf.to_file(dest_file, driver='GPKG')
        logger.debug(f"Created editable GPkg for {county} at {dest_file}")
        return dest_file

    except PreventUpdate:
        raise
    except Exception as e:
        logger.error(f"Error copying GPkg for county {county}: {e}")
        raise PreventUpdate


    
def standardize_county_name(name):
    """
    Standardize county names by removing 'County' suffix and ensuring proper casing.

    Parameters:
        name (str): Original county name.

    Returns:
        str: Standardized county name.
    """
    if pd.isna(name):
        return 'Unknown'
    name = name.strip().title()
    if name.endswith(' County'):
        name = name.replace(' County', '')
    return name

def load_data_final(file_path):
    """
    Load and preprocess Data_Final.csv, including Crash_Date, Crash_Time, 
    Data_Type, Crash_Type, and SeverityCategory.
    """
    usecols = [
        'CaseNumber',            # → Case_Number
        'CrashDate',             # → Crash_Date
        'CrashTimeFormatted',    # → Crash_Time
        'RoadSurfaceCondition',  # → RoadSurfac
        'WeatherCondition',      # → WeatherCon
        'LightCondition',        # → LightCon
        'CountyName',            # → County
        'CrashCategory',         # → Data_Type
        'CrashType',             # → Crash_Type
        'Longitude',             # → X_Coord
        'Latitude',              # → Y_Coord
        'SeverityCategory'       
    ]

    dtype = {
        'CaseNumber': str,
        'CrashDate': str,
        'CrashTimeFormatted': str,
        'RoadSurfaceCondition': str,
        'WeatherCondition': str,
        'LightCondition': str,
        'CountyName': str,
        'CrashCategory': str,
        'CrashType': str,
        'Longitude': float,
        'Latitude': float,
        'SeverityCategory': str     
    }

    chunks = []
    try:
        for i, chunk in enumerate(pd.read_csv(
            file_path,
            usecols=usecols,
            dtype=dtype,
            chunksize=100_000,
            header=0
        )):
            # Rename standard schema
            chunk = chunk.rename(columns={
                'CaseNumber': 'Case_Number',
                'CrashDate': 'Crash_Date',
                'CrashTimeFormatted': 'Crash_Time',
                'RoadSurfaceCondition': 'RoadSurfac',
                'WeatherCondition': 'WeatherCon',
                'LightCondition': 'LightCon',
                'CrashCategory': 'Data_Type',
                'CrashType': 'Crash_Type',
                'Longitude': 'X_Coord',
                'Latitude': 'Y_Coord',
                'CountyName': 'County',
                'SeverityCategory': 'SeverityCategory'  
            })

            # Standardize County
            chunk['County'] = chunk['County'].fillna('Unknown').apply(standardize_county_name)

            # Parse dates & times
            chunk['Crash_Date'] = pd.to_datetime(chunk['Crash_Date'], errors='coerce')
            chunk['Crash_Time'] = (
                pd.to_datetime(chunk['Crash_Time'], format='%I:%M %p', errors='coerce')
                  .dt.hour
            )

            # Fill/clean the string columns
            chunk['WeatherCon'] = chunk['WeatherCon'].fillna('Unknown').str.title()
            chunk['LightCon']   = chunk['LightCon'].fillna('Unknown').str.title()
            chunk['RoadSurfac'] = chunk['RoadSurfac'].fillna('Unknown').str.title()
            chunk['Data_Type']  = chunk['Data_Type'].fillna('Non-VRU').str.upper()
            chunk['Crash_Type'] = chunk['Crash_Type'].fillna('Unknown').str.upper()

            # Reorder so SeverityCategory is at the end
            chunk = chunk[
                ['Case_Number','X_Coord','Y_Coord','Crash_Date','Crash_Time',
                 'WeatherCon','LightCon','RoadSurfac','Data_Type',
                 'County','Crash_Type','SeverityCategory']
            ]

            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=[
            'Case_Number','X_Coord','Y_Coord','Crash_Date','Crash_Time',
            'WeatherCon','LightCon','RoadSurfac','Data_Type',
            'County','Crash_Type','SeverityCategory'
        ])

        logger.debug(f"Loaded {len(df)} records from Data_Final.csv (with SeverityCategory).")
        return df

    except Exception as e:
        logger.error(f"Error loading Data_Final.csv: {e}")
        return pd.DataFrame(columns=[
            'Case_Number','X_Coord','Y_Coord','Crash_Date','Crash_Time',
            'WeatherCon','LightCon','RoadSurfac','Data_Type',
            'County','Crash_Type','SeverityCategory'
        ])


def load_all_data_optimized(data_final_file, counties):
    """
    Optimized loading of Data_Final.csv, splitting data by county.

    Parameters:
        data_final_file (str): Path to Data_Final.csv.
        counties (list): List of counties to load data for.

    Returns:
        dict: A dictionary with county names as keys and combined DataFrames as values.
    """
    data_by_county = {county: pd.DataFrame(columns=[
        'Case_Number', 'X_Coord', 'Y_Coord', 'Crash_Date',
        'Crash_Time', 'WeatherCon', 'LightCon', 'RoadSurfac', 'Data_Type', 'County'
    ]) for county in counties}

    # Load and preprocess Data_Final.csv once
    data_final_df = load_data_final(data_final_file)
    for county in counties:
        # Filter Data_Final data by county
        county_data_final = data_final_df[data_final_df['County'] == county]
        if county_data_final.empty:
            logger.warning(f"No Data_Final data found for county: {county}")
        else:
            logger.debug(f"Data_Final data for county {county}: {len(county_data_final)} records.")
        data_by_county[county] = county_data_final

    return data_by_county

# Mapping of COUNTYFP to county names in New York State
county_fips_map = {
    '001': 'Albany',
    '003': 'Allegany',
    '005': 'Bronx',
    '007': 'Broome',
    '009': 'Cattaraugus',
    '011': 'Cayuga',
    '013': 'Chautauqua',
    '015': 'Chemung',
    '017': 'Chenango',
    '019': 'Clinton',
    '021': 'Columbia',
    '023': 'Cortland',
    '025': 'Delaware',
    '027': 'Dutchess',
    '029': 'Erie',
    '031': 'Essex',
    '033': 'Franklin',
    '035': 'Fulton',
    '037': 'Genesee',
    '039': 'Greene',
    '041': 'Hamilton',
    '043': 'Herkimer',
    '045': 'Jefferson',
    '047': 'Kings',
    '049': 'Lewis',
    '051': 'Livingston',
    '053': 'Madison',
    '055': 'Monroe',
    '057': 'Montgomery',
    '059': 'Nassau',
    '061': 'New York',
    '063': 'Niagara',
    '065': 'Oneida',
    '067': 'Onondaga',
    '069': 'Ontario',
    '071': 'Orange',
    '073': 'Orleans',
    '075': 'Oswego',
    '077': 'Otsego',
    '079': 'Putnam',
    '081': 'Queens',
    '083': 'Rensselaer',
    '085': 'Richmond',
    '087': 'Rockland',
    '089': 'St. Lawrence',
    '091': 'Saratoga',
    '093': 'Schenectady',
    '095': 'Schoharie',
    '097': 'Schuyler',
    '099': 'Seneca',
    '101': 'Steuben',
    '103': 'Suffolk',
    '105': 'Sullivan',
    '107': 'Tioga',
    '109': 'Tompkins',
    '111': 'Ulster',
    '113': 'Warren',
    '115': 'Washington',
    '117': 'Wayne',
    '119': 'Westchester',
    '121': 'Wyoming',
    '123': 'Yates',
}

def load_census_data(file_path):
    """
    Load and preprocess census data from a GeoPackage (.gpkg) file.
    Normalize CNTY_NAME to strip the ' County' suffix and title-case.
    """
    census_polygons_by_county = {}
    try:
        gdf = gpd.read_file(file_path)
        for idx, row in gdf.iterrows():
            raw = row.get('CNTY_NAME')
            if pd.isna(raw):
                continue

            # strip off any trailing " County" and title-case
            county_name = standardize_county_name(raw)

            # pick up the geometry
            geom = row.get('geom') or row.get('geometry')
            if geom is None:
                continue

            poly = mapping(geom)
            props = row.to_dict()
            props.pop('geom', None)
            props.pop('geometry', None)
            poly['properties'] = props

            census_polygons_by_county.setdefault(county_name, []).append(poly)

        return census_polygons_by_county

    except Exception as e:
        logger.error(f"Error loading Census data: {e}")
        return {}


# ----------------------------
# 3. Initialize Dash App
# ----------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'])
app.title = 'Crash Data Analytics'
server = app.server

# Initialize caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple'  
})

# Define cache timeout (e.g., 1 hour)
CACHE_TIMEOUT = 60 * 60

# ----------------------------
# 4. Define Filter Functions
# ----------------------------


def convert_miles_to_pixels(miles, zoom, center_latitude):
    """
    Convert a distance in miles to a pixel radius for a density mapbox.

    Parameters:
        miles (float): The desired radius in miles.
        zoom (float): The current zoom level of the map.
        center_latitude (float): The latitude of the map center.

    Returns:
        float: The corresponding radius in pixels.
    """
    meters_per_mile = 1609.34
    # Convert miles to meters.
    distance_meters = miles * meters_per_mile
    # Calculate meters per pixel at the given zoom level and latitude.
    meters_per_pixel = 156543.03392 * math.cos(math.radians(center_latitude)) / (2 ** zoom)
    # Return the radius in pixels.
    pixel_radius = distance_meters / meters_per_pixel
    return pixel_radius

def filter_data_tab1(df, start_date, end_date, time_range, days_of_week,
                     weather, light, road_surface,
                     severity_category, crash_type,
                     main_data_type, vru_data_type):    
    if start_date and end_date:
        df = df[(df['Crash_Date'] >= start_date) & (df['Crash_Date'] <= end_date)]
    # Time filtering
    if time_range:
        df = df[(df['Crash_Time'] >= time_range[0]) & (df['Crash_Time'] <= time_range[1])]
    # Day of week filtering
    if days_of_week:
        df = df[df['Crash_Date'].dt.day_name().isin(days_of_week)]
    # Weather, Light, and Road Surface filtering
    if weather != 'All':
        df = df[df['WeatherCon'] == weather]
    if light != 'All':
        df = df[df['LightCon'] == light]
    if road_surface != 'All':
        df = df[df['RoadSurfac'] == road_surface]
    if severity_category != 'All':
        df = df[df['SeverityCategory'] == severity_category]
    if crash_type and crash_type != 'All':       
       df = df[df['Crash_Type'].str.strip().str.upper() == crash_type.upper()]

    
    # Crash type filtering based on main_data_type:
    if main_data_type == 'VRU':
        # Filter to only VRU-related crashes.
        if vru_data_type == 'ALL_VRU':
            df = df[df['Crash_Type'].str.strip().str.upper().isin(
                ['COLLISION WITH BICYCLIST', 'COLLISION WITH PEDESTRIAN']
            )]
        elif vru_data_type == 'BICYCLE':
            df = df[df['Crash_Type'].str.strip().str.upper() == 'COLLISION WITH BICYCLIST']
        elif vru_data_type == 'PEDESTRIAN':
            df = df[df['Crash_Type'].str.strip().str.upper() == 'COLLISION WITH PEDESTRIAN']
    elif main_data_type == 'All':
        # "All" returns every crash (both VRU and non-VRU).
        pass 
    elif main_data_type == 'None':
        # "None" returns no crash data.
        df = df.iloc[0:0]
    
    return df


# ----------------------------
# 5. Define UI Components
# ----------------------------

def common_controls(prefix, show_buttons, available_counties, unique_weather, unique_light, unique_road , unique_crash_types):
    controls = [
        html.Div([
            html.Label('County:'),
            html.Div(
                "i",
                title='Select one or more counties. Selecting multiple counties can degrade performance.',
                style={
                    'display': 'inline-block',
                    'background-color': '#ccc',
                    'border': '1px solid #999',
                    'border-radius': '3px',
                    'width': '20px',
                    'height': '20px',
                    'text-align': 'center',
                    'line-height': '20px',
                    'cursor': 'help',
                    'margin-left': '5px'
                }
            )
        ]),
        dcc.Dropdown(
            id=f'county_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': county, 'value': county} for county in available_counties],
            value=['Albany'],  # default county
            multi=True,
            placeholder='Select one or more counties or All',
            style={'width': '100%'}  # let the container control the overall width
        ),
        html.Label('Data Type:', style={'margin-top': '10px'}),
        dcc.RadioItems(
            id=f'data_type_selector_main_{prefix}',
            options=[
                {'label': 'All', 'value': 'All'},
                {'label': 'VRU', 'value': 'VRU'}
            ],
            value='All',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
        ),
        # This div is hidden unless "VRU" is selected above.
        html.Div(
            id=f'data_type_vru_options_{prefix}',
            children=[
                dcc.RadioItems(
                    id=f'data_type_selector_vru_{prefix}',
                    options=[
                        {'label': 'All', 'value': 'ALL_VRU'},
                        {'label': 'Bicycle', 'value': 'BICYCLE'},
                        {'label': 'Pedestrian', 'value': 'PEDESTRIAN'}
                    ],
                    value='ALL_VRU',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                )
            ],
            style={'display': 'none'}
        ),
        html.Label('Date Range:', style={'margin-top': '10px'}),
        dcc.DatePickerRange(
            id=f'date_picker_{prefix}',
            start_date='2020-01-01',
            end_date='2023-12-31',
            display_format='YYYY-MM-DD',
            start_date_placeholder_text='Select a start date',
            style={
                'transform': 'scale(0.8)',
                'transform-origin': 'top left',
                'font-size': '12px',
                'padding': '2px',
                'display': 'inline-block',
                'margin-top': '5px'
            }
        ),
        html.Label('Time Range (Hour):', style={'margin-top': '20px'}),
        dcc.RangeSlider(
            id=f'time_slider_{prefix}',
            min=0,
            max=23,
            step=1,
            value=[0, 6],
            marks={
                0: '12am', 3: '3am', 6: '6am', 9: '9am', 12: '12pm', 
                15: '3pm', 18: '6pm', 21: '9pm', 23: '11pm'
            },
            tooltip={'always_visible': True}
        ),
        html.Label('Days of the Week:', style={'margin-top': '20px'}),
        dcc.Checklist(
            id=f'day_of_week_checklist_{prefix}',
            options=[
                {'label': 'Mon', 'value': 'Monday'},
                {'label': 'Tue', 'value': 'Tuesday'},
                {'label': 'Wed', 'value': 'Wednesday'},
                {'label': 'Thu', 'value': 'Thursday'},
                {'label': 'Fri', 'value': 'Friday'},
                {'label': 'Sat', 'value': 'Saturday'},
                {'label': 'Sun', 'value': 'Sunday'}
            ],
            value=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            labelStyle={'display': 'inline-block', 'margin-right': '5px', 'font-size': '12px'},
            style={'margin-top': '10px'}
        ),
        html.Label('Select Weather Condition:', style={'margin-top': '20px'}),
        dcc.Dropdown(
            id=f'weather_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': w, 'value': w} for w in unique_weather],
            value='All',
            placeholder='Select weather condition',
            style={'width': '100%'}
        ),
        html.Label('Select Road Surface Condition:', style={'margin-top': '20px'}),
        dcc.Dropdown(
            id=f'road_surface_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': road, 'value': road} for road in unique_road],
            value='All',
            placeholder='Select road surface condition',
            style={'width': '100%'}
        ),
        html.Label('Select Light Condition:', style={'margin-top': '20px'}),
        dcc.Dropdown(
            id=f'light_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': l, 'value': l} for l in unique_light],
            value='All',
            placeholder='Select light condition',
            style={'width': '100%'}
        ),
        
        html.Label('Select Crash Type:', style={'margin-top': '20px'}),
        dcc.Dropdown(
            id=f'crash_type_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': ct,    'value': ct} for ct in unique_crash_types],
            value='All',
            placeholder='Select crash type',
            style={'width': '100%'}
        ),

        html.Label('Select Severity Category:', style={'margin-top': '20px'}),
        dcc.Dropdown(
            id=f'severity_selector_{prefix}',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': v, 'value': v} for v in ['Fatal', 'Non-Fatal']],
            value='All',
            placeholder='Select severity category',
            style={'width': '100%'}
        ),
    ]
    
    if show_buttons:
        if prefix == 'tab1':
            controls += [
                html.Div([
                    html.Button('Apply Filter', id=f'apply_filter_{prefix}', n_clicks=0, style={'margin-top': '30px'}),
                    html.Button('Clear Drawing', id=f'clear_drawing_{prefix}', n_clicks=0, style={'margin-top': '10px', 'margin-left': '10px'}),
                    html.Button('Download Filtered Data', id='download_button_tab1', n_clicks=0, style={'margin-top': '10px', 'display': 'block'})
                ])
            ]
        else:
            controls += [
                html.Div([
                    html.Button('Apply Filter', id=f'apply_filter_{prefix}', n_clicks=0, style={'margin-top': '30px'})
                ])
            ]
    
    # Remove the inline width, float, and margin settings and use a CSS class instead.
    return html.Div(controls, className='responsive-controls')

def census_controls():
    counties = sorted(census_polygons_by_county.keys())

    controls = [
        html.Div([
            html.Label('County:'),
            html.Div(
                "i",
                title='Select one or more counties. Use "All" to include every county.',
                style={
                    'display': 'inline-block',
                    'background-color': '#ccc',
                    'border': '1px solid #999',
                    'border-radius': '3px',
                    'width': '20px',
                    'height': '20px',
                    'text-align': 'center',
                    'line-height': '20px',
                    'cursor': 'help',
                    'margin-left': '5px'
                }
            )
        ]),
        dcc.Dropdown(
            id='county_selector_tab3',
            options=[{'label': 'All', 'value': 'All'}] +
                    [{'label': c, 'value': c} for c in counties],
            value=['Albany'],  # default selection
            multi=True,
            placeholder='Select one or more counties or All',
            style={'width': '100%'}
        ),
        html.Label("Select Census Attribute for Opacity:", style={'margin-top': '20px'}),
        dcc.Dropdown(
            id='census_attribute_selector',
            options=[
                {'label': 'Demographic Index',               'value': 'DEMOGIDX_5'},
                {'label': 'People of Color %',               'value': 'PEOPCOLORPCT'},
                {'label': 'Unemployment %',                  'value': 'UNEMPPCT'},
                {'label': 'Residential %',                   'value': 'pct_residential'},
                {'label': 'Industrial %',                    'value': 'pct_industrial'},
                {'label': 'Retail %',                        'value': 'pct_retail'},
                {'label': 'Commercial %',                    'value': 'pct_commercial'},
                {'label': 'AADT Crash Rate',                 'value': 'AADT Crash Rate'},
                {'label': 'VRU Crash Rate',                  'value': 'VRU Crash Rate'},
                {'label': 'AADT',                            'value': 'AADT'},
                {'label': 'Commute TripMiles Start Avg',     'value': 'Commute_TripMiles_TripStart_avg'},
                {'label': 'Commute TripMiles End Avg',       'value': 'Commute_TripMiles_TripEnd_avg'},
                {'label': 'Commute Biking and Walking Mile', 'value': 'Commute_BIKING_and_WALKING_Mile'},
                {'label': 'Commute Biking and Walking Mi 1', 'value': 'Commute_BIKING_and_WALKING_Mi_1'},
            ],
            value='DEMOGIDX_5',
            style={'width': '100%'}
        ),
        html.Div(id='census_color_legend', style={'margin-top': '20px'}),
        # contextual notes
        html.Div([
            html.P("Crash data 2020-2023", style={'margin': '0'}),
            html.P("Environmental justice data 2023", style={'margin': '0'}),
            html.P("Census tract data 2023", style={'margin': '0'}),
            html.P("Poverty data 2022", style={'margin': '0'}),
            html.P("Disability data 2022", style={'margin': '0'}),
            html.P("Race/Pop data 2022", style={'margin': '0'})
        ], style={
            'margin-top': '10px',
            'font-size': '12px',
            'color': '#333',
            'border': '1px solid #ccc',
            'padding': '5px'
        }),
        html.Div([
            html.Button('Apply Filter', id='apply_filter_tab3', n_clicks=0, style={'margin-top': '30px'})
        ], style={'margin-bottom': '20px', 'margin-top': '20px'})
    ]

    return html.Div(controls, className='responsive-controls')

# ----------------------------
# 6. Define the Main Layout
# ----------------------------
app.layout = html.Div([
    # Navigation Tabs
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Data Downloader', value='tab-1'),
        dcc.Tab(label='Heatmap', value='tab-2'),
        dcc.Tab(label='Census Data', value='tab-3'),
        dcc.Tab(label='Predictions', value='tab-4'),
        dcc.Tab(label='ChatBot', value='tab-5'),
    ], style={'position': 'fixed', 'top': '0', 'left': '0', 'width': '100%', 'zIndex': '1000'}),

    # Header Section
    html.Div([
        html.Div([
            html.Img(src='/assets/Poly.svg', style={
                'height': '128px', 'float': 'left', 'margin-right': '40px', 
                'margin-left': '-20px', 'margin-top': '-8px'
            }),
            html.H1('Crash Data Analytics', className='app-title'),
            html.Img(src='/assets/NY.svg', className='ny-logo')
        ],style={
            'backgroundColor': '#18468B', 'padding': '7.5px', 'position': 'fixed', 
            'top': '50px', 'left': '0', 'width': '100%', 'zIndex': '999', 'height': '90px'
        }),

        # Dynamic Content Based on Selected Tab
        html.Div([
            html.Div(id='tabs-content', style={'margin-top': '160px'})
        ]),
    ]),

    # Download Components
    dcc.Download(id='download_data'),
    dcc.Download(id='download_data_tab3'),   
    
    dcc.Store(id='editable_gpkg_path'),
    dcc.Store(id='selected_census_tract'),
    dcc.Store(id='predictions_refresh', data=0),
    html.Button(
        id='refresh_predictions_tab4',
        n_clicks=0,
        style={'display': 'none'}
    ),
    
    html.Div(
    html.Button('Edit Selected Census Tract', id='open_edit_modal', n_clicks=0),
    style={'display': 'none'}
),


# Modal for editing county data:
html.Div(
    id='county_edit_modal',
    children=[
        html.H3("Edit County Data"),
        html.Div(
            id="modal_fields_container",
            children=[
                make_field_row(var, LABELS[var], STEPS[var])
                for var, _, _ in ALL_FIELDS
            ]
        ),
        html.Div([
            html.Button(
                "Apply Updated Data",
                id="apply_updated_data",
                n_clicks=0,
                style={'marginRight': '10px'}
            ),
            #unused but do not remove or else everything breaks (i keep removing it)
            html.Button(
                "Reset Predictions",
                id="reset_predictions",
                n_clicks=0,
                style={'marginRight': '10px'}
            ),
            html.Button(
                "Close",
                id="close_modal",
                n_clicks=0
            ),
        ], style={'marginBottom': '10px', 'textAlign': 'center'})
    ],
    style={
        'display': 'none',
        'position': 'fixed',
        'top': '50%',
        'left': '50%',
        'transform': 'translate(-50%, -50%)',
        'padding': '20px',
        'backgroundColor': 'white',
        'border': '2px solid black',
        'zIndex': 1000
    }
)



])

# ----------------------------
# 7. Define Callbacks
# ----------------------------

# Global variable to store data by county
data_by_county = {}

# Define county coordinates globally for access in callbacks
county_coordinates = {}

# Cache memoization
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

message = "Hello! I am an interactive safety chatbot designed to provide you with real-time, data-driven insights on roadway safety. Whether you seek information about high-risk areas, traffic incident trends, or general road safety guidance, I will offer reliable and context-aware responses.\n\n" \
            "**Example Prompts**\n\n" \
            "- What are the top 5 cities with the most crashes in 2021, showing counts?\n\n" \
            "- What is the average number of injuries for crashes involving a commercial vehicle?\n\n" \
            "- Describe a typical crash involving a pedestrian.\n\n" \
            "- Plot all pedestrian-related crashes in Buffalo. \n\n"
# Callback to render content based on selected tab
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    available_counties = list(county_coordinates.keys())
    
    # Ensure that the unique lists are available (they must be defined globally after loading the data)
    global unique_weather, unique_light, unique_road

    if tab == 'tab-1': # Data Downloader Tab
        return html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            common_controls(
                                'tab1',
                                show_buttons=True,
                                available_counties=available_counties,
                                unique_weather=unique_weather,
                                unique_light=unique_light,
                                unique_road=unique_road,
                                unique_crash_types=unique_crash_types
                            ),
                            className='responsive-controls'
                        ),
                        dcc.Graph(
                            id='scatter_map',
                            className='responsive-graph',
                            figure={
                                'data': [],
                                'layout': {
                                    'mapbox': {
                                        'style': "open-street-map",
                                        'center': {
                                            'lat': county_coordinates[available_counties[0]]['lat'],
                                            'lon': county_coordinates[available_counties[0]]['lon']
                                        },
                                        'zoom': 10
                                    },
                                    'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}
                                }
                            },
                            config={'modeBarButtonsToRemove': ['lasso2d'], 'displayModeBar': True, 'scrollZoom': True}
                        )
                    ],
                    className='desktop-layout'
                ),
                html.Div(
                    id='warning_message_tab1',
                    style={'color': 'red', 'textAlign': 'center', 'margin': '10px'}
                )
            ]
        )

    elif tab == 'tab-2': # Heatmap Tab
        return html.Div([
            html.Div([
                html.Div([
                    html.Label('Adjust Heatmap Radius:', style={'font-weight': 'bold'}),
                    html.Div(
                        "i",
                        title="This slider adjusts the radius of influence for the heatmap.",
                        style={
                            'display': 'inline-block',
                            'background-color': '#ccc',
                            'border': '1px solid #999',
                            'border-radius': '3px',
                            'width': '20px',
                            'height': '20px',
                            'text-align': 'center',
                            'line-height': '20px',
                            'cursor': 'help',
                            'margin-left': '5px'
                        }
                    ),
                    dcc.Slider(
                        id='radius_slider_tab2',
                        min=0.1,
                        max=10,
                        step=0.1,
                        value=1,
                        marks={
                            0.1: '0.1 mi',
                            1: '1 mi',
                            2: '2 mi',
                            3: '3 mi',
                            4: '4 mi',
                            5: '5 mi',
                            6: '6 mi',
                            7: '7 mi',
                            8: '8 mi',
                            9: '9 mi',
                            10: '10 mi'
                        },
                        tooltip={'placement': 'bottom', 'always_visible': True}
                    )
                ], style={'margin-bottom': '20px'}),
                
                common_controls(
                    'tab2',
                    show_buttons=True,
                    available_counties=available_counties,
                    unique_weather=unique_weather,
                    unique_light=unique_light,
                    unique_road=unique_road,
                    unique_crash_types=unique_crash_types)
            ], className='responsive-controls'),
            
            # Right-side: The Heatmap Graph container
            html.Div([
                dcc.Graph(
                    id='heatmap_graph',
                    className='responsive-graph',
                    figure={
                        'data': [],
                        'layout': {
                            'mapbox': {
                                'style': "open-street-map",
                                'center': {
                                    'lat': county_coordinates[available_counties[0]]['lat'],
                                    'lon': county_coordinates[available_counties[0]]['lon']
                                },
                                'zoom': 10
                            },
                            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}
                        }
                    },
                    config={'modeBarButtonsToRemove': ['lasso2d'], 'displayModeBar': True, 'scrollZoom': True}
                ),
                # Hidden button for closing the popup (must exist in initial layout)
                html.Button(html.I(className="fa-window-close"),id="close-popup-button", n_clicks=0, style={'display': 'none'}),

                dcc.Loading(
                    html.Div(id='image-popup', style={
                        'position': 'fixed',
                        'left': '50%',
                        'top': '50%',
                        'transform': 'translate(-50%, -50%)',
                        'zIndex': '1000',
                        'backgroundColor': 'white',
                        'border': '1px solid black',
                        'padding': '10px',
                        'display': 'none', # Start hidden
                        'maxWidth': '320px',
                        'boxShadow': '0px 0px 10px rgba(0,0,0,0.5)'
                    }),
                ),
                html.Button(
                    id='insight-button',n_clicks=0, 
                    children=[
                        html.I(className="fas fa-lightbulb", title="Get AI Powered Insights"),
                    ]),
        ], className='responsive-graph'),
    
    ], className='desktop-layout')

    elif tab == 'tab-3': # Census Data Tab
        return html.Div([
            html.Div([
                census_controls(),
                html.Div(id='warning_message_tab3', style={'color': 'red'})
            ], className='responsive-controls'),
            html.Div(
                dcc.Graph(
                    id='scatter_map_tab3',
                    className='responsive-graph',
                    figure={
                        'data': [],
                        'layout': {
                            'mapbox': {
                                'style': "open-street-map",
                                'center': {
                                    'lat': county_coordinates[available_counties[0]]['lat'],
                                    'lon': county_coordinates[available_counties[0]]['lon']
                                },
                                'zoom': 10
                            },
                            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}
                        }
                    },
                    config={
                        'modeBarButtonsToRemove': ['lasso2d'], 
                        'displayModeBar': True, 
                        'scrollZoom': True
                    }
                ),
                className='responsive-graph'
            )
        ], className='desktop-layout')

    elif tab == 'tab-4':  # Predictions Tab
        return html.Div([
            # Left-side: Controls (Model, Prediction Bar, County Selector, and Editing UI)
            html.Div([
                html.Div([
                    html.Label('Select AI Model:', style={'fontSize': '12px'}),
                    html.Span(
                        "i",
                        title="Choose a tract to edit and apply the new values, then reset to change another one. Update one tract at a time.",
                        style={
                            'display': 'inline-block',
                            'backgroundColor': '#ccc',
                            'border': '1px solid #999',
                            'borderRadius': '50%',
                            'width': '16px',
                            'height': '16px',
                            'textAlign': 'center',
                            'lineHeight': '16px',
                            'cursor': 'help',
                            'marginLeft': '5px',
                            'fontSize': '12px'
                        }
                    )
                ], style={'display': 'flex', 'alignItems': 'center'}),
                dcc.Dropdown(
                    id='model_selector_tab4',
                    options=[
                        {'label': 'MGWR Model',   'value': 'mgwr_predict.py'},
                    ],
                    value='mgwr_predict.py',
                    clearable=False,
                    style={'width': '100%', 'fontSize': '12px'}
                ),
                html.Div(
                    id='prediction_bar',
                    style={
                        'width': '100%',
                        'padding': '10px',
                        'backgroundColor': '#eee',
                        'textAlign': 'center',
                        'fontWeight': 'bold'
                    }
                ),
                dcc.Store(id='original_prediction'),
                html.Br(),
                html.Label('Select County:', style={'fontSize': '12px'}),
                dcc.Dropdown(
                    id='county_selector_tab4',
                    options=[],  # to be updated via callback
                    multi=True,
                    placeholder='Select county by CNTY_NAME',
                    style={'width': '100%', 'fontSize': '12px'}
                ),
                html.Br(),
                html.Label('Prediction Data Controls', style={'fontSize': '12px'}),
                html.Button('Refresh Predictions', id='refresh_predictions_tab4', n_clicks=0, style={'fontSize': '12px'}),
                html.Hr(),
                html.Div(
                    [ make_field_row(var, LABELS[var], STEPS[var]) for var,_,_ in ALL_FIELDS ],
                    id="modal_fields_container"
                ),
                html.Div([
                html.Button("Apply Updated Data",
                    id="apply_updated_data",
                    n_clicks=0,
                    style={'marginRight':'10px','fontSize':'12px'}),
                html.Button("Reset Predictions",
                    id="reset_predictions",
                    n_clicks=0,
                    style={'fontSize':'12px'})
                    ], style={'marginTop':'10px','textAlign':'center'}),
            ], className='responsive-controls'),
            # Right-side: Predictions Map
            html.Div([
                # Predictions map
                dcc.Graph(
                    id='predictions_map',
                    className='responsive-graph',
                    figure={
                        'data': [],
                        'layout': {
                            'mapbox': {
                                'style': "open-street-map",
                                'center': {'lat': 40.7128, 'lon': -74.0060},
                                'zoom': 10
                            },
                            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0}
                        }
                    },
                    config={
                        'modeBarButtonsToRemove': ['lasso2d'],
                        'displayModeBar': True,
                        'scrollZoom': True
                    }
                ),

                # Comparison scatter plot
                dcc.Graph(
                    id='comparison_graph',
                    className='responsive-graph'
                )
            ], className='responsive-graph'),
        ], className='desktop-layout')
    
    elif tab == 'tab-5': # Chatbot Tab
        return chatbotlayout.load_chatbot_layout([{"sender": "bot", "message": message, "map": None}])

@app.callback(
    Output('chat-history-store', 'data'),
    Output('chat-history-container', 'children'),
    Input('clear-button', 'n_clicks'),
    State('chat-history-store', 'data'),
    State('chat-history-container', 'children'),
    prevent_initial_call=True
)
def clear_chat_history(n_clicks, current_chat_data, curent_chat_container):
    if n_clicks and n_clicks > 0:
        # Reset the chat history to an empty list
        # while len(chat_history) > 1:
        #     chat_history.pop()
        return current_chat_data[0:1], curent_chat_container[0:1] # Empty list for store, empty list for children
    return dash.no_update, dash.no_update # If button not clicked, do nothing

@app.callback(
    Output('image-popup', 'children'),
    Output('image-popup', 'style'),
    Output('heatmap_graph', 'clickData'),
    Input('heatmap_graph', 'clickData'),
    Input('insight-button', 'n_clicks'), 
    Input('close-popup-button', 'n_clicks'),
    State('heatmap_graph', 'figure'),      
    prevent_initial_call=True
)
def manage_popup_display(clickData, insight_button_n_clicks, close_button_n_clicks, fig_snapshot):
    triggered_id = ctx.triggered_id

    # If the close button was clicked
    if triggered_id == 'close-popup-button' and close_button_n_clicks is not None:
        print("Close button clicked, hiding popup.")
        return None, {'display': 'none'}, None

    if triggered_id == 'insight-button' and insight_button_n_clicks and insight_button_n_clicks > 0:
        fig_width = 1280
        fig_height = 720
        #pio.write_image(fig_snapshot, "density_map.png", scale=1, width=fig_width, height=fig_height)
        image_bytes = pio.to_image(fig=fig_snapshot, format='png', scale=1, width=fig_width, height=fig_height)
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        data_url = f"data:image/png;base64,{encoded_image}"
        insights = call_with_timeout(map_analyzer.generate_response, 30, image_bytes)
        popup_style = {
            'display': 'block',
            'position': 'fixed',
            'left': '50%',
            'top': '50%',
            'transform': 'translate(-50%, -50%)',
            'zIndex': '1000',
            'backgroundColor': 'white',
            'border': '1px solid black',
            'padding': '10px',
            'maxWidth': '1280px',
            'maxHeight': '720px',
            'boxShadow': '0px 0px 10px rgba(0,0,0,0.5)'
        }
        image_element = html.Div([
            html.H1("Image Insights"),
            html.Button(html.I(className="fa fa-window-close"), id="close-popup-button", n_clicks=0),
            dcc.Markdown(insights, style={'maxWidth': '640px'}),
            html.Img(src=data_url, style={'maxWidth': '640px', 'maxHeight': '480px'})
        ])
        return image_element, popup_style, None
    
    if triggered_id == 'heatmap_graph':
        print(f"Click data {clickData}.")    
        lon = clickData['points'][0]['lon']
        lat = clickData['points'][0]['lat']
        print(f"Lon: {lon}, Lat: {lat}")
        location_name = f"Lon: {lon:.4f}, Lat: {lat:.4f}"
        if 'customdata' in clickData['points'][0]:
            crash_info = clickData['points'][0]['customdata']
            crash_data_dict = {
                'Latitude': lat,
                'Longitude': lon,
                'CaseNumber': crash_info[0],
                'CrashDate': crash_info[1],
                'CrashTime': crash_info[2],
                'WeatherCondition': crash_info[3],
                'LightCondition': crash_info[4],
                'RoadSurfaceCondition': crash_info[5]
                }
        else:
            crash_data_dict = {
                'Latitude': lat,
                'Longitude': lon,
            }
        print(crash_data_dict)
        image = call_with_timeout(streetview.get_street_view_image, 30, lat, lon)
        analysis = call_with_timeout(streetview.analyze_image_ai, 30, image.content, crash_data_dict)

        popup_style = {
            'display': 'block',
            'position': 'fixed',
            'left': '50%',
            'top': '50%',
            'transform': 'translate(-50%, -50%)',
            'zIndex': '1000',
            'backgroundColor': 'white',
            'border': '1px solid black',
            'padding': '10px',
            'maxWidth': '1280px',
            'boxShadow': '0px 0px 10px rgba(0,0,0,0.5)'
        }
        image_element = html.Div([
            html.H1(location_name),
            html.Button(html.I(className="fa fa-window-close"), id="close-popup-button", n_clicks=0),
            dcc.Markdown(analysis, style={'maxWidth': '540px'}),
            html.Img(src=image.url, style={'maxWidth': '640px', 'maxHeight': '640px'})
        ])
        print(f"Displaying popup for {location_name}.")
        return image_element, popup_style, None

    # Fallback or initial state
    print("No valid trigger for display/hide, returning no_update.")
    return dash.no_update, dash.no_update, None

# --- Python Callback 1: Handle User Input and Display Immediately (with loading placeholder) ---
@app.callback(
    Output('user-input', 'value'),
    Output('chat-history-store', 'data', allow_duplicate=True),
    Output('scroll-trigger', 'data', allow_duplicate=True),
    Output('user-question-for-bot', 'data'),
    [Input('send-button', 'n_clicks'),
     Input('user-input', 'n_submit')],
    State('user-input', 'value'),
    State('chat-history-store', 'data'),
    State('scroll-trigger', 'data'),
    prevent_initial_call=True
)
def handle_user_input(send_button_clicks, n_submits, user_question, current_chat_data, current_scroll_trigger):
    if not user_question or user_question.strip() == "":
        raise dash.exceptions.PreventUpdate

    # Append user message
    msg = {"sender": "user", "message": user_question}
    current_chat_data.append(msg)
    #chat_history.append(msg)

    # Append temporary loading message
    loading_msg = {"sender": "bot", "message": "Thinking...", "map": None}
    current_chat_data.append(loading_msg)

    new_scroll_trigger = current_scroll_trigger + 1

    return (
        '',
        current_chat_data, 
        new_scroll_trigger, 
        {
            "question": user_question,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    )

# --- Python Callback 2: Generate Bot Response (updates the specific bot message) ---
@app.callback(
    Output('chat-history-store', 'data', allow_duplicate=True),
    Output('scroll-trigger', 'data', allow_duplicate=True),    
    Input('user-question-for-bot', 'data'),
    State('chat-history-store', 'data'),
    State('scroll-trigger', 'data'),
    prevent_initial_call=True
)
def generate_and_display_bot_response(user_question_data, current_chat_data, current_scroll_trigger):
    if user_question_data is None:
        raise dash.exceptions.PreventUpdate

    user_question = user_question_data["question"]

    #bot_response_message_content = chatbotv1.generate_response(user_question)
    bot_response_data = chatbotv4.get_agent_response(user_question)
    bot_response_text = bot_response_data.get("text", "No response.")
    fig = bot_response_data.get("visualization_data")
    
    # Remove loading message
    current_chat_data.pop()

    msg = {"sender": "bot", "message": bot_response_text, "map": fig}
    current_chat_data.append(msg)
    #chat_history.append(msg)
    new_scroll_trigger = current_scroll_trigger + 1

    return current_chat_data, new_scroll_trigger

# --- Python Callback 3: Update Chat History Display and Scroll after all data is in chat-history-store ---
@app.callback(
    Output('chat-history-container', 'children', allow_duplicate=True),
    Output('chat-history-store', 'data', allow_duplicate=True),
    Output('scroll-trigger', 'data', allow_duplicate=True),
    Input('chat-history-store', 'data'), # Listen to changes in the history store
    prevent_initial_call='initial_duplicate'
)
def update_chat_display(stored_chat_data):
    if stored_chat_data is None:
        raise dash.exceptions.PreventUpdate
    
    rendered_history_elements =[]
    for msg in stored_chat_data:
        if msg['sender'] == "user":
            rendered_history_elements.append(chatbotlayout.render_user_message_bubble(msg['message']))
        elif msg['sender'] == "bot":
            rendered_history_elements.append(chatbotlayout.render_bot_message_bubble(msg['message'], msg['map']))
    #rendered_history_elements = [chatbotlayout.render_message_bubble(msg['sender'], msg['message']) for msg in stored_chat_data]
    rendered_history_elements.append(html.Div(id='chat-end-marker'))
    return rendered_history_elements, stored_chat_data, 0

# --- Clientside Callback for Auto-Scrolling ---
clientside_callback(
    """
    function(data) {
        // This function is triggered by the 'scroll-trigger' data change
        // It needs to be robust, so it only attempts to scroll if the marker exists.
        const marker = document.getElementById('chat-end-marker');
        if (marker) {
            marker.scrollIntoView({ behavior: 'smooth' }); // 'smooth' for animated scroll
        }
        return window.dash_clientside.no_update; // Don't update any Dash output
    }
    """,
    Output('scroll-trigger', 'data', allow_duplicate=True), # Dummy output to trigger the clientside callback
    Input('scroll-trigger', 'data'),   # Input is the data from our Python callback
    prevent_initial_call=True, # Prevent scrolling on initial page load from this callback
)

@app.callback(
    Output('data_type_vru_options_tab1', 'style'),
    Input('data_type_selector_main_tab1', 'value')
)
def toggle_vru_options_tab1(main_value):
    return {'display': 'block'} if main_value == 'VRU' else {'display': 'none'}

@app.callback(
    Output('data_type_vru_options_tab2', 'style'),
    Input('data_type_selector_main_tab2', 'value')
)
def toggle_vru_options_tab2(main_value):
    return {'display': 'block'} if main_value == 'VRU' else {'display': 'none'}

@app.callback(
    Output('data_type_vru_options_tab3', 'style'),
    Input('data_type_selector_main_tab3', 'value')
)
def toggle_vru_options_tab3(main_value):
    return {'display': 'block'} if main_value == 'VRU' else {'display': 'none'}



# ----------------------------
# 7.1. Callback for Data Downloader Tab (tab1)
# ----------------------------
@app.callback(
    Output('comparison_graph', 'figure'),
    Input('predictions_refresh',  'data'),
    State('model_selector_tab4',  'value'),
    State('county_selector_tab4', 'value'),
    State('editable_gpkg_path',   'data'),
)
def update_comparison_graph(refresh, model_file, selected_counties, editable_gpkg_path):
    # choose file‐suffix + default global file
    if model_file == "AI2.py":
        suffix, default_file = "_with_gwr_predictions", DEFAULT_PRED_FILES['AI2.py']
    elif model_file == "mgwr_predict.py":
        suffix, default_file = "_with_mgwr_predictions", DEFAULT_PRED_FILES['mgwr_predict.py']
    else:
        suffix, default_file = "_with_predictions",     DEFAULT_PRED_FILES['AI.py']

    # only splitext when editable_gpkg_path is actually a string
    if isinstance(editable_gpkg_path, str) and editable_gpkg_path:
        base, ext = os.path.splitext(editable_gpkg_path)
        candidate = f"{base}{suffix}{ext}"
        gpkg_file = candidate if os.path.exists(candidate) else default_file
    else:
        gpkg_file = default_file

    # load the GeoPackage
    gdf = gpd.read_file(gpkg_file)

    # apply county filter if requested
    if selected_counties:
        if 'CNTY_NAME' in gdf.columns:
            gdf['CNTY_NAME'] = (
                gdf['CNTY_NAME']
                   .str.replace(" County", "", regex=False)
                   .str.strip()
                   .str.title()
            )
            gdf = gdf[gdf['CNTY_NAME'].isin(selected_counties)]
        elif 'CountyName' in gdf.columns:
            gdf['CountyName'] = (
                gdf['CountyName']
                   .str.strip()
                   .str.title()
            )
            gdf = gdf[gdf['CountyName'].isin(selected_counties)]

    # ensure we have the “Prediction” column
    if 'Prediction' not in gdf.columns:
        return go.Figure()

    # build comparison scatter
    fig = go.Figure()
    if 'AADT Crash Rate' in gdf.columns:
        fig.add_trace(go.Scatter(
            x=gdf['Prediction'],
            y=gdf['AADT Crash Rate'],
            mode='markers',
            name='AADT Crash Rate',
            hovertemplate=(
                "<b>AADT Crash Rate</b><br>"
                "Prediction: %{x:.2f}<br>"
                "Observed: %{y:.2f}<extra></extra>"
            )
        ))
    if 'VRU Crash Rate' in gdf.columns:
        fig.add_trace(go.Scatter(
            x=gdf['Prediction'],
            y=gdf['VRU Crash Rate'],
            mode='markers',
            name='VRU Crash Rate',
            hovertemplate=(
                "<b>VRU Crash Rate</b><br>"
                "Prediction: %{x:.2f}<br>"
                "Observed: %{y:.2f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Model Prediction vs. Observed Crash Rates",
        xaxis_title="Prediction",
        yaxis_title="Crash Rate",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


@app.callback(
    [
        Output('scatter_map', 'figure'),
        Output('scatter_map', 'selectedData')
    ],
    [
        Input('apply_filter_tab1', 'n_clicks'),
        Input('clear_drawing_tab1', 'n_clicks'),
    ],
    [
        State('county_selector_tab1', 'value'),
        State('scatter_map', 'selectedData'),
        State('date_picker_tab1', 'start_date'),
        State('date_picker_tab1', 'end_date'),
        State('time_slider_tab1', 'value'),
        State('day_of_week_checklist_tab1', 'value'),
        State('weather_selector_tab1', 'value'),
        State('light_selector_tab1', 'value'),
        State('road_surface_selector_tab1', 'value'),
        State('severity_selector_tab1','value'),
        State('data_type_selector_main_tab1', 'value'),
        State('data_type_selector_vru_tab1', 'value'),
        State('crash_type_selector_tab1','value')
    ]
)
def map_tab1(apply_n_clicks, clear_n_clicks, counties_selected, selected_data,
             start_date, end_date, time_range, days_of_week,
            weather, light, road_surface, severity_category, main_data_type, vru_data_type, crash_type):
    ctx = callback_context
    triggered = (
        ctx.triggered[0]['prop_id'].split('.')[0]
        if ctx.triggered else 'initial_load'
    )
    logger.debug(f"Triggered Input: {triggered}")

    # load data
    df = get_county_data(counties_selected)

    # compute default center
    if isinstance(counties_selected, list) and counties_selected and 'All' not in counties_selected:
        lat_center = sum(county_coordinates[c]['lat'] for c in counties_selected) / len(counties_selected)
        lon_center = sum(county_coordinates[c]['lon'] for c in counties_selected) / len(counties_selected)
    else:
        lat_center, lon_center = 40.7128, -74.0060

    # helper to set uirevision key
    key = 'tab1-' + '-'.join(sorted(counties_selected or []))

    # empty‐data fallback
    if df.empty:
        fig = px.scatter_map(
            pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
            lat='Latitude', lon='Longitude', zoom=10, map_style="open-street-map"
        )
        fig.update_traces(marker=dict(opacity=0))
        fig.update_layout(uirevision=key)
        return fig, None

    # apply filters
    if triggered == 'apply_filter_tab1':
        filtered = filter_data_tab1(
            df, start_date, end_date, time_range,
            days_of_week, weather, light, road_surface,  severity_category, crash_type,
            main_data_type, vru_data_type, 
        )
        if selected_data and 'points' in selected_data:
            keep = [pt['customdata'][0] for pt in selected_data['points']]
            filtered = filtered[filtered['Case_Number'].isin(keep)]
        df_to_plot = filtered
        out_selected = selected_data
        if crash_type and crash_type != 'All':
            df_to_plot = df_to_plot[df_to_plot['Crash_Type'] == crash_type]

    elif triggered == 'clear_drawing_tab1':
        # reapply filters but drop box selection
        df_to_plot = filter_data_tab1(
            df, start_date, end_date, time_range,
            days_of_week, weather, light, road_surface, severity_category,
            main_data_type, vru_data_type, crash_type
        )
        out_selected = None

    else:  # initial_load
        df_to_plot = filter_data_tab1(
            df, '1900-01-01', '1901-01-01', [0, 23],
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
            'All','All','All', 'All', 'All',
            main_data_type, vru_data_type
        )
        out_selected = None

    # build the figure
    if df_to_plot.empty:
        fig = px.scatter_map(
            pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
            lat='Latitude', lon='Longitude', zoom=10, map_style="open-street-map"
        )
        fig.update_traces(marker=dict(opacity=0))
    else:
        fig = px.scatter_map(
            df_to_plot,
            lat='Y_Coord', lon='X_Coord', zoom=10, map_style="open-street-map",
            hover_name='Case_Number',
            hover_data={
                'Crash_Date': True, 'Crash_Time': True,
                'WeatherCon': True, 'LightCon': True,
                'RoadSurfac': True
            },
            custom_data=['Case_Number']
        )
        fig.update_layout(mapbox_center={'lat': lat_center, 'lon': lon_center})

    fig.update_layout(uirevision=key)
    return fig, out_selected


# Callback to Download Filtered Data in Data Downloader Tab (tab1)
@app.callback(
    Output('download_data', 'data'),
    [Input('download_button_tab1', 'n_clicks')],
    [
        State('county_selector_tab1', 'value'),
        State('date_picker_tab1', 'start_date'),
        State('date_picker_tab1', 'end_date'),
        State('time_slider_tab1', 'value'),
        State('day_of_week_checklist_tab1', 'value'),
        State('weather_selector_tab1', 'value'),
        State('light_selector_tab1', 'value'),
        State('road_surface_selector_tab1', 'value'),
        State('data_type_selector_main_tab1', 'value'),
        State('data_type_selector_vru_tab1', 'value'),
        State('scatter_map', 'selectedData')
    ]
)
def download_filtered_data_tab1(n_clicks, counties_selected, start_date, end_date, time_range, days_of_week,
                                weather, light, road_surface, main_data_type, vru_data_type, selected_data):
    if n_clicks > 0:
        try:
            # Load full data for the selected counties.
            df = get_county_data(counties_selected)
            if df.empty:
                logger.warning(f"No data available to download for counties: {counties_selected}")
                raise PreventUpdate

            # Apply the same filters as in update_map_tab1.
            filtered_df = filter_data_tab1(
                df, start_date, end_date, time_range,
                days_of_week, weather, light, road_surface,
                main_data_type, vru_data_type
            )

            # If a box selection exists, further filter by the selected points.
            if selected_data and 'points' in selected_data and selected_data['points']:
                selected_case_numbers = [point['customdata'][0] for point in selected_data['points']]
                filtered_df = filtered_df[filtered_df['Case_Number'].isin(selected_case_numbers)]
                logger.debug(f"Downloading {len(filtered_df)} records after box selection filtering.")
            else:
                logger.debug(f"Downloading all {len(filtered_df)} records after applying filters.")

            return dcc.send_data_frame(filtered_df.to_csv, filename="filtered_data.csv")
        except Exception as e:
            logger.error(f"Error in download_filtered_data_tab1: {e}")
            raise PreventUpdate
    return None


# ----------------------------
# 7.2. Callback for Heatmap Tab (tab2)
# ----------------------------

def filter_data_tab2(df, data_type):
    """
    Filter data based on data type for Heatmap Tab.

    Parameters:
        df (pd.DataFrame): The DataFrame to filter.
        data_type (str): 'All' or 'VRU'.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if data_type != 'All':
        df = df[df['Data_Type'] == data_type]
    return df

@app.callback(
    Output('heatmap_graph', 'figure'),
    Input('apply_filter_tab2', 'n_clicks'),
    State('radius_slider_tab2', 'value'),
    State('county_selector_tab2', 'value'),
    State('data_type_selector_main_tab2', 'value'),
    State('data_type_selector_vru_tab2', 'value'),
    State('severity_selector_tab2', 'value'),
    State('date_picker_tab2', 'start_date'),
    State('date_picker_tab2', 'end_date'),
    State('time_slider_tab2', 'value'),
    State('day_of_week_checklist_tab2', 'value'),
    State('weather_selector_tab2', 'value'),
    State('light_selector_tab2', 'value'),
    State('road_surface_selector_tab2', 'value'),
    State('crash_type_selector_tab2','value')
)
def update_heatmap_tab2(n_clicks, radius_miles, counties_selected,
                        main_data_type, vru_data_type,
                        severity_category,
                        start_date, end_date, time_range,
                        days_of_week, weather, light, road_surface, crash_type):
    zoom = 10
    default_center = {'lat': 40.7128, 'lon': -74.0060}
    key = 'tab2-' + '-'.join(sorted(counties_selected or []))

    # BEFORE FIRST CLICK: show a blank‐centered “dot”
    if not n_clicks:
        radius_px = convert_miles_to_pixels(radius_miles, zoom, default_center['lat'])
        fig = px.density_map(
            pd.DataFrame(default_center, index=[0]),
            lat='lat', lon='lon',
            radius=radius_px,
            center=default_center, zoom=zoom,
            map_style="open-street-map", opacity=0.7
        )
        fig.update_layout(uirevision=key)
        return fig

    # LOAD YOUR DATA
    df = get_county_data(counties_selected)
    if df.empty:
        # same blank‐dot fallback
        radius_px = convert_miles_to_pixels(radius_miles, zoom, default_center['lat'])
        fig = px.density_map(
            pd.DataFrame(default_center, index=[0]),
            lat='lat', lon='lon',
            radius=radius_px,
            center=default_center, zoom=zoom,
            map_style="open-street-map", opacity=0.7
        )
        fig.update_layout(uirevision=key)
        return fig

    # APPLY *exactly* the same filters as Tab1, *including* VRU sub‐type
    filtered = filter_data_tab1(
        df,
        start_date, end_date, time_range,
        days_of_week, weather, light, road_surface,
        severity_category, crash_type,
        main_data_type, vru_data_type, 
    )

    # COMPUTE CENTER
    if filtered.empty:
        center_lat, center_lon = default_center.values()
    elif 'All' in counties_selected:
        center_lat, center_lon = df['Y_Coord'].mean(), df['X_Coord'].mean()
    else:
        pts = [county_coordinates[c] for c in counties_selected]
        center_lat = sum(p['lat'] for p in pts) / len(pts)
        center_lon = sum(p['lon'] for p in pts) / len(pts)


    # RE‐COMPUTE PIXEL RADIUS AT NEW CENTER
    radius_px = convert_miles_to_pixels(radius_miles, zoom, center_lat)

    # BUILD THE DENSITY MAP
    fig = px.density_map(
        filtered,
        lat='Y_Coord', lon='X_Coord',
        radius=radius_px,
        center={'lat': center_lat, 'lon': center_lon},
        zoom=zoom,
        map_style="open-street-map",
        opacity=0.5,
        hover_data={
            'Case_Number': True,
            'Crash_Date': True,
            'Crash_Time': True,
            'WeatherCon': True,
            'LightCon': True,
            'RoadSurfac': True
        }
    )
    fig.update_layout(uirevision=key)
    return fig



# ----------------------------
# 7.3. Callback for Census Data Tab (tab3)
# ----------------------------
@app.callback(
    Output('scatter_map_tab3', 'figure'),
    Input('apply_filter_tab3', 'n_clicks'),
    State('county_selector_tab3', 'value'),
    State('census_attribute_selector', 'value')
)
def update_map_tab3(apply_n_clicks, counties_selected, selected_attribute):
    # Build a uirevision key based on the selected counties
    key = 'tab3-' + '-'.join(sorted(counties_selected or []))

    fig = go.Figure()

    # Determine which counties to draw
    if isinstance(counties_selected, list) and 'All' in counties_selected:
        selected_counties = list(census_polygons_by_county.keys())
    else:
        selected_counties = counties_selected or []

    # Gather values for normalization
    vals = []
    for county in selected_counties:
        for poly in census_polygons_by_county.get(county, []):
            try:
                vals.append(float(poly["properties"].get(selected_attribute, 0)))
            except:
                pass
    min_val, max_val = (min(vals), max(vals)) if vals else (0, 1)

    # Plot each polygon with opacity based on its value
    for county in selected_counties:
        for poly in census_polygons_by_county.get(county, []):
            coords_list = []
            geoms = [poly.get('coordinates')] if poly.get('type') == 'Polygon' else poly.get('coordinates', [])
            for coords in geoms:
                ring = coords[0] if poly.get('type') == 'Polygon' else coords[0]
                if ring[0] != ring[-1]:
                    ring = ring + [ring[0]]
                if len(ring) < 3:
                    continue
                lons, lats = zip(*ring)
                try:
                    norm = (float(poly["properties"].get(selected_attribute, 0)) - min_val) / (max_val - min_val)
                except:
                    norm = 0.5
                opacity = 0.1 + 0.9 * norm
                fig.add_trace(go.Scattermapbox(
                    lat=list(lats),
                    lon=list(lons),
                    mode='lines',
                    fill='toself',
                    fillcolor=f'rgba(0,255,0,{opacity})',
                    line=dict(color='green', width=2),
                    hoverinfo='skip',
                    showlegend=False
                ))

    # Compute center
    all_lats = [pt for trace in fig.data for pt in trace.lat]
    all_lons = [pt for trace in fig.data for pt in trace.lon]
    if all_lats and all_lons:
        center_lat, center_lon = sum(all_lats)/len(all_lats), sum(all_lons)/len(all_lons)
    else:
        # fallback to first county or NYC
        fallback = county_coordinates.get(selected_counties[0], {'lat': 40.7128, 'lon': -74.0060})
        center_lat, center_lon = fallback['lat'], fallback['lon']

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center={'lat': center_lat, 'lon': center_lon},
            zoom=10
        ),
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        uirevision=key
    )

    return fig


# Callback to Download Filtered Data in Census Data Tab (tab3)
@app.callback(
    Output('download_data_tab3', 'data'),
    [Input('download_button_tab3', 'n_clicks')],
    [
        State('scatter_map_tab3', 'figure'),
        State('county_selector_tab3', 'value')
    ]
)
def download_filtered_data_tab3(n_clicks, figure, counties_selected):
    """
    Download filtered data from Census Data Tab.
    """
    if n_clicks > 0:
        try:
            df = get_county_data(counties_selected)
            if df.empty:
                logger.warning(f"No data available to download for counties: {counties_selected}")
                raise PreventUpdate

            if figure and 'data' in figure and figure['data']:
                selected_case_numbers = []
                for trace in figure['data']:
                    if trace.name.endswith(" Census Polygon"):
                        continue
                    if 'customdata' in trace:
                        for point in trace.customdata:
                            selected_case_numbers.append(point[0])
                if selected_case_numbers:
                    filtered_df = df[df['Case_Number'].isin(selected_case_numbers)]
                    logger.debug(f"Downloading {len(filtered_df)} selected records for Census Data in counties: {counties_selected}")
                else:
                    filtered_df = df
                    logger.debug(f"Downloading all {len(filtered_df)} records for Census Data in counties: {counties_selected}")
            else:
                filtered_df = df

            logger.debug(f"Downloading {len(filtered_df)} records for Census Data in counties: {counties_selected}")
            return dcc.send_data_frame(filtered_df.to_csv, filename="census_filtered_data.csv")
        except Exception as e:
            logger.error(f"Error in download_filtered_data_tab3: {e}")
            raise PreventUpdate
    return None

@app.callback(
    Output('census_color_legend', 'children'),
    [Input('apply_filter_tab3', 'n_clicks')],
    [
        State('county_selector_tab3', 'value'),
        State('census_attribute_selector', 'value')
    ]
)
def update_color_legend(n_clicks, counties_selected, selected_attribute):
    # Determine selected counties.
    if isinstance(counties_selected, list):
        if 'All' in counties_selected:
            selected_counties = list(census_polygons_by_county.keys())
        else:
            selected_counties = counties_selected
    else:
        selected_counties = [counties_selected]

    # Gather all attribute values.
    all_attr_values = []
    for county in selected_counties:
        polygons = census_polygons_by_county.get(county, [])
        if not polygons:
            alternate_key = county + " County"
            polygons = census_polygons_by_county.get(alternate_key, [])
        for poly in polygons:
            try:
                val = float(poly["properties"].get(selected_attribute, 0))
                all_attr_values.append(val)
            except (TypeError, ValueError):
                pass

    if all_attr_values:
        min_val = min(all_attr_values)
        max_val = max(all_attr_values)
    else:
        min_val, max_val = 0, 1

    # Create a gradient legend using a CSS linear-gradient.
    legend = html.Div([
        html.Div("Low: {:.2f}".format(min_val), style={'float': 'left', 'fontSize': '12px'}),
        html.Div("High: {:.2f}".format(max_val), style={'float': 'right', 'fontSize': '12px'}),
        html.Div(style={
            'clear': 'both',
            'height': '20px',
            'background': 'linear-gradient(to right, rgba(0,255,0,0.1), rgba(0,255,0,1))',
            'margin-top': '5px'
        })
    ], style={'margin-top': '10px', 'border': '1px solid #ccc', 'padding': '5px'})
    return legend




# Global variable to store mapping from polygon id to county name.
county_mapping = {}
@app.callback(
    Output('predictions_map', 'figure'),
    [
        Input('refresh_predictions_tab4', 'n_clicks'),
        Input('county_selector_tab4', 'value'),
        Input('predictions_refresh', 'data'),
        Input('model_selector_tab4', 'value')
    ],
    State('editable_gpkg_path', 'data')
)
def update_predictions_map(n_clicks, selected_counties, refresh_trigger, model_file, editable_gpkg_path):
    # build a uirevision key so that changing only predictions doesn't move the map
    counties_key = '-'.join(sorted(selected_counties or []))
    key = f"tab4-{model_file}-{counties_key}"

    # pick which gpkg & column
    if model_file == "AI2.py":
        suffix, pred_col, default_file = "_with_gwr_predictions",  "GWR_Prediction", DEFAULT_PRED_FILES['AI2.py']
    elif model_file == "mgwr_predict.py":
        suffix, pred_col, default_file = "_with_mgwr_predictions", "MGWR_Prediction", DEFAULT_PRED_FILES['mgwr_predict.py']
    else:
        suffix, pred_col, default_file = "_with_predictions",     "Prediction",      DEFAULT_PRED_FILES['AI.py']

    # decide which file to load (only split if we actually have a string path)
    if isinstance(editable_gpkg_path, str) and editable_gpkg_path:
        base, ext = os.path.splitext(editable_gpkg_path)
        candidate = f"{base}{suffix}{ext}"
        gpkg_file = candidate if os.path.exists(candidate) else default_file
    else:
        gpkg_file = default_file


    try:
        gdf = gpd.read_file(gpkg_file)
        if pred_col not in gdf.columns:
            raise KeyError(f"Missing '{pred_col}' in {gpkg_file}")
        gdf['Prediction'] = gdf[pred_col]

        # normalize county names & filter
        # ——— filter by selected_counties, using whichever county field exists ———
        if selected_counties:
            # AI/GWR outputs
            if 'CNTY_NAME' in gdf.columns:
                gdf['CNTY_NAME'] = (
                    gdf['CNTY_NAME']
                       .str.replace(" County", "", regex=False)
                       .str.strip()
                       .str.title()
                )
                gdf = gdf[gdf['CNTY_NAME'].isin(selected_counties)]

            # MGWR outputs use CountyName
            elif 'CountyName' in gdf.columns:
                gdf['CountyName'] = (
                    gdf['CountyName']
                       .str.replace(" County", "", regex=False)
                       .str.strip()
                       .str.title()
                )
                gdf = gdf[gdf['CountyName'].isin(selected_counties)]

            # else: no county column, so skip filtering entirely

        valid   = gdf[~gdf['Prediction'].isna()]
        missing = gdf[ gdf['Prediction'].isna()]

        fig = go.Figure([
            go.Choroplethmapbox(
                geojson=json.loads(missing.to_json()),
                locations=missing['id'],
                z=[0]*len(missing),
                colorscale=[[0,"black"],[1,"black"]],
                marker_opacity=0.9,
                marker_line_width=1,
                showscale=False,
                featureidkey="properties.id",
                name="Missing",
                hovertemplate="<extra></extra>" 
            ),
                go.Choroplethmapbox(
                geojson=json.loads(valid.to_json()),
                locations=valid['id'],
                z=valid['Prediction'],
                colorscale='YlGnBu',
                marker_opacity=0.6,
                marker_line_width=1,
                colorbar=dict(title="Prediction"),
                featureidkey="properties.id",
                name="Prediction",
                hovertemplate=
                "<b>Tract %{location}</b><br>" +
                "Predicted Crash Rate: %{z:.2f}<extra></extra>"
            )
        ])

        # compute center only if there's data
        if not gdf.empty:
            ctr = gdf.geometry.centroid
            center = {'lat': ctr.y.mean(), 'lon': ctr.x.mean()}
        else:
            center = {'lat': 40.7128, 'lon': -74.0060}

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=center,
                zoom=10
            ),
            margin={'l':0,'r':0,'t':0,'b':0},
            legend=dict(x=0, y=1),
            uirevision=key
        )
        return fig

    except Exception as e:
        logger.error(f"Error in update_predictions_map: {e}", exc_info=True)
        # fallback blank map, also preserving camera if possible
        fig = go.Figure(go.Scattermapbox(
            lat=[40.7128], lon=[-74.0060],
            mode='markers', marker=dict(opacity=0)
        ))
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center={'lat':40.7128,'lon':-74.0060},
                zoom=10
            ),
            margin={'l':0,'r':0,'t':0,'b':0},
            annotations=[dict(
                text="Error loading predictions map",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False
            )],
            uirevision=key
        )
        return fig


# -----------------------------------------------------------------------------
# 1) update_modal_values: reload on tract/model/refresh/gpkg change, else bump values
# -----------------------------------------------------------------------------
# 1) Grab your field IDs from ALL_FIELDS:
FIELD_IDS = [ var for var, _, _ in ALL_FIELDS ]

# 2) Build the decorator args:
modal_value_outputs = [
    Output(f"input_{var}", "value")
    for var in FIELD_IDS
]

modal_value_inputs = (
    [ Input("selected_census_tract", "data") ]
  + sum([[ 
       Input(f"plus_input_{var}",  "n_clicks"),
       Input(f"minus_input_{var}", "n_clicks")
    ] for var in FIELD_IDS], [])
)

modal_value_states = (
    [ State("editable_gpkg_path", "data") ]
  + [ State(f"input_{var}", "value") for var in FIELD_IDS ]
)

@app.callback(
    *modal_value_outputs,
    *modal_value_inputs,
    *modal_value_states,
    prevent_initial_call=True
)
def update_modal_values(*all_args):
    """
    On tract‐click: load every field from the editable GPKG.
    On plus/minus: bump only that one field by its step, leave the rest unchanged.
    """
    num = len(FIELD_IDS)
    # Inputs are: 1 selected_tract + 2*num clicks
    trigger_args = all_args[: 1 + 2 * num]
    # States are: 1 gpkg_path + num current values
    state_args   = all_args[1 + 2 * num :]

    selected_tract = trigger_args[0]
    gpkg_path      = state_args[0]
    current_vals   = list(state_args[1:])  # length == num

    trig = callback_context.triggered[0]["prop_id"]

    def clean(v):
        return None if pd.isna(v) else round(v, 2)

    # --- Case 1: new tract selected → load all fields from the GPKG
    if trig.startswith("selected_census_tract"):
        if not (selected_tract and gpkg_path and os.path.exists(gpkg_path)):
            raise PreventUpdate
        gdf = gpd.read_file(gpkg_path)
        rename_map = {
            v.replace('(','.').replace(')','.'): v
            for v in FIELD_IDS
        }
        gdf = gdf.rename(columns=rename_map)
        row = gdf[gdf["id"].astype(str) == str(selected_tract)]
        if row.empty:
            raise PreventUpdate
        row = row.iloc[0]
        return [ clean(row.get(var)) for var in FIELD_IDS ]

    # --- Case 2: plus/minus clicked → find exactly which var to adjust
    # Map each plus/minus input to its index and sign:
    deltas = {}
    for idx, var in enumerate(FIELD_IDS):
        if trig.startswith(f"plus_input_{var}.n_clicks"):
            deltas[idx] = STEPS[var]
        elif trig.startswith(f"minus_input_{var}.n_clicks"):
            deltas[idx] = -STEPS[var]

    if not deltas:
        # no recognized trigger → do nothing
        raise PreventUpdate

    # apply the single delta and return all values
    new_vals = current_vals.copy()
    for idx, delta in deltas.items():
        base = new_vals[idx] or 0
        new_vals[idx] = round(base + delta, 2)

    return new_vals

@app.callback(
    Output('county_selector_tab4', 'options'),
    [
        Input('refresh_predictions_tab4', 'n_clicks'),
        Input('model_selector_tab4',    'value')
    ]
)
def update_county_options(n_clicks, model_file):
    try:
        if model_file == "AI2.py":
            gpkg_file = DEFAULT_PRED_FILES['AI2.py']
        elif model_file == "MGWR.py":
            gpkg_file = DEFAULT_PRED_FILES['MGWR.py']
        else:
            gpkg_file = DEFAULT_PRED_FILES['AI.py']
        gdf = gpd.read_file(gpkg_file)
        # Log the original column values for debugging
        logger.debug("Original CNTY_NAME values: " + str(gdf['CNTY_NAME'].unique()))
        # Remove the " County" suffix and standardize
        gdf['CNTY_NAME'] = gdf['CNTY_NAME'].str.replace(" County", "", regex=False).str.strip().str.title()
        unique_counties = sorted(gdf['CNTY_NAME'].dropna().unique().tolist())
        logger.debug("Unique counties after formatting: " + str(unique_counties))
        options = [{'label': county, 'value': county} for county in unique_counties]
        return options
    except Exception as e:
        logger.error(f"Error updating county selector options: {e}")
        return []



@app.callback(
    Output('county_edit_modal', 'style'),
    [Input('open_edit_modal', 'n_clicks'),
     Input('close_modal', 'n_clicks'),
     Input('apply_updated_data', 'n_clicks')],
    State('county_edit_modal', 'style')
)
def toggle_modal(open_clicks, close_clicks, apply_clicks, current_style):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'open_edit_modal':
        # Show the modal with styling consistent with your project.
        return {
            'display': 'block',
            'position': 'fixed',
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)',
            'padding': '20px',
            'backgroundColor': 'white',
            'border': '2px solid black',
            'zIndex': 1000
        }
    else:
        # Hide the modal on Close or after applying changes.
        return {'display': 'none'}


@app.callback(
    Output('selected_census_tract', 'data'),
    Output('editable_gpkg_path',    'data'),
    Output('predictions_refresh',   'data'),
    Output('county_selector_tab4',  'value'),
    Input('predictions_map',        'clickData'),
    Input('county_selector_tab4',   'value'),
    Input('apply_updated_data',     'n_clicks'),
    Input('reset_predictions',      'n_clicks'),
    Input('model_selector_tab4',    'value'),
    State('selected_census_tract',  'data'),
    State('editable_gpkg_path',     'data'),
    # all 24 field‐States in the right order
    State('input_DEMOGIDX_5',                        'value'),
    State('input_PEOPCOLORPCT',                      'value'),
    State('input_UNEMPPCT',                          'value'),
    State('input_pct_residential',                   'value'),
    State('input_pct_industrial',                    'value'),
    State('input_pct_retail',                        'value'),
    State('input_pct_commercial',                    'value'),
    State('input_AADT',                              'value'),
    State('input_Commute_TripMiles_TripStart_avg',   'value'),
    State('input_Commute_TripMiles_TripEnd_avg',     'value'),
    State('input_ACSTOTPOP',                         'value'),
    State('input_DEMOGIDX_2',                        'value'),
    State('input_PovertyPop',                        'value'),
    State('input_DISABILITYPCT',                     'value'),
    State('input_BikingTrips(Start)',                'value'),
    State('input_BikingTrips(End)',                  'value'),
    State('input_CarpoolTrips(Start)',               'value'),
    State('input_CarpoolTrips(End)',                 'value'),
    State('input_CommercialFreightTrips(Start)',     'value'),
    State('input_CommercialFreightTrips(End)',       'value'),
    State('input_WalkingTrips(Start)',               'value'),
    State('input_WalkingTrips(End)',                 'value'),
    State('input_PublicTransitTrips(Start)',         'value'),
    State('input_PublicTransitTrips(End)',           'value'),
    State('predictions_refresh',                     'data'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def manage_editable_and_predictions(
    clickData,
    county_val,
    apply_n, reset_n, model_file,
    selected_tract, gpkg_path,
    demogidx_5, peopcolorpct, unemppct,
    pct_residential, pct_industrial, pct_retail, pct_commercial,
    aadt, commute_start, commute_end,
    acstotpop, demogidx_2, poverty_pop, disabilitypct,
    biking_start, biking_end,
    carpool_start, carpool_end,
    freight_start, freight_end,
    walking_start, walking_end,
    transit_start, transit_end,
    current_refresh
):



    trig = ctx.triggered_id

    # 1) Map click just update selected_census_tract
    if trig == 'predictions_map':
        if clickData and clickData.get('points'):
            tract = clickData['points'][0]['location']
            return tract, dash.no_update, dash.no_update, dash.no_update
        raise PreventUpdate


    # Model swapclear everything so user must re-choose county
    if trig == 'model_selector_tab4':
        # delete any existing editable & pred files but only if gpkg_path is really a string
        if isinstance(gpkg_path, str) and os.path.exists(gpkg_path):
            os.remove(gpkg_path)
            base, ext = os.path.splitext(gpkg_path)
            suffix = (
                '_with_gwr_predictions'   if model_file == 'AI2.py' else
                '_with_mgwr_predictions'  if model_file == 'mgwr_predict.py' else
                '_with_predictions'
            )
            pred_p = f"{base}{suffix}{ext}"
            if os.path.exists(pred_p):
                os.remove(pred_p)
        # clear out both the county dropdown and the tract store
        return None, None, (current_refresh or 0) + 1, []

    # 3) New county selected  copy per county file
    if trig == 'county_selector_tab4':
        if county_val and len(county_val)==1:
            cnty = county_val[0]
            if model_file=='AI2.py':
                src, dst = DEFAULT_PRED_FILES['AI2.py'],    './AI/'
            elif model_file=='mgwr_predict.py':
                src, dst = DEFAULT_PRED_FILES['mgwr_predict.py'],'./MGWR/'
            else:
                src, dst = DEFAULT_PRED_FILES['AI.py'],     './AI/'
            new_path = copy_county_gpkg(cnty, src, dst)
            return (
          dash.no_update,     # selected_tract
          new_path,           # editable_gpkg_path
          dash.no_update,     # predictions_refresh
          dash.no_update      # county_selector itself
        )
        raise PreventUpdate

    # 4) Apply edits → write GPKG, rerun model, bump
    if trig == 'apply_updated_data':
        if apply_n and selected_tract and gpkg_path:
            gdf = gpd.read_file(gpkg_path)
            idx = gdf[gdf['id']==selected_tract].index
            if idx.empty:
                raise PreventUpdate

            # — your 24 field‐writes —
            gdf.loc[idx, 'DEMOGIDX_5']                      = demogidx_5
            gdf.loc[idx, 'PEOPCOLORPCT']                    = peopcolorpct
            gdf.loc[idx, 'UNEMPPCT']                        = unemppct
            gdf.loc[idx, 'pct_residential']                 = pct_residential
            gdf.loc[idx, 'pct_industrial']                  = pct_industrial
            gdf.loc[idx, 'pct_retail']                      = pct_retail
            gdf.loc[idx, 'pct_commercial']                  = pct_commercial
            gdf.loc[idx, 'AADT']                            = aadt
            gdf.loc[idx, 'Commute_TripMiles_TripStart_avg'] = commute_start
            gdf.loc[idx, 'Commute_TripMiles_TripEnd_avg']   = commute_end
            gdf.loc[idx, 'ACSTOTPOP']                       = acstotpop
            gdf.loc[idx, 'DEMOGIDX_2']                      = demogidx_2
            gdf.loc[idx, 'PovertyPop']                      = poverty_pop
            gdf.loc[idx, 'DISABILITYPCT']                   = disabilitypct
            gdf.loc[idx, 'BikingTrips(Start)']              = biking_start
            gdf.loc[idx, 'BikingTrips(End)']                = biking_end
            gdf.loc[idx, 'CarpoolTrips(Start)']             = carpool_start
            gdf.loc[idx, 'CarpoolTrips(End)']               = carpool_end
            gdf.loc[idx, 'CommercialFreightTrips(Start)']   = freight_start
            gdf.loc[idx, 'CommercialFreightTrips(End)']     = freight_end
            gdf.loc[idx, 'WalkingTrips(Start)']             = walking_start
            gdf.loc[idx, 'WalkingTrips(End)']               = walking_end
            gdf.loc[idx, 'PublicTransitTrips(Start)']       = transit_start
            gdf.loc[idx, 'PublicTransitTrips(End)']         = transit_end

            # overwrite and re‐run
            gdf.to_file(gpkg_path, driver="GPKG")
            base, ext = os.path.splitext(gpkg_path)
            suffix   = '_with_mgwr_predictions' if model_file=='mgwr_predict.py' else '_with_predictions'
            out_file = f"{base}{suffix}{ext}"
            subprocess.run([sys.executable, model_file, gpkg_path, out_file],
                           check=True, capture_output=True, text=True)

            return dash.no_update, dash.no_update, (current_refresh or 0) + 1, dash.no_update

        raise PreventUpdate

    # Reset button (delete & recopy or it breaks)
    if trig == 'reset_predictions' and reset_n:
        # delete editable + pred only if gpkg_path is a valid path
        if isinstance(gpkg_path, str) and os.path.exists(gpkg_path):
            os.remove(gpkg_path)
            base, ext = os.path.splitext(gpkg_path)
            suffix = '_with_mgwr_predictions' if model_file == 'mgwr_predict.py' else '_with_predictions'
            pred_p = f"{base}{suffix}{ext}"
            if os.path.exists(pred_p):
                os.remove(pred_p)

        # immediately recopy so user can click again
        new_path = None
        if county_val and len(county_val) == 1:
            cnty = county_val[0]
            if model_file == 'AI2.py':
                src, dst = DEFAULT_PRED_FILES['AI2.py'],    './AI/'
            elif model_file == 'mgwr_predict.py':
                src, dst = DEFAULT_PRED_FILES['mgwr_predict.py'], './MGWR/'
            else:
                src, dst = DEFAULT_PRED_FILES['AI.py'],     './AI/'
            new_path = copy_county_gpkg(cnty, src, dst)

            return (dash.no_update,  new_path, (current_refresh or 0) + 1, dash.no_update)

    # fallback if it all breaks
    raise PreventUpdate



@app.callback(
    [Output(f"container_{var}", "style") for var, _, _ in ALL_FIELDS],
    [
        Input("model_selector_tab4",       "value"),
        Input("selected_census_tract",     "data"),
        Input("editable_gpkg_path",        "data"),
    ],
    prevent_initial_call=True,
)
def toggle_field_styles(model_file, selected_tract, gpkg_path):
    # AI and GWR always show all fields (can add more here later)
    if model_file in ("AI.py", "AI2.py"):
        return [{'marginBottom': '10px'}] * len(ALL_FIELDS)

    # MGWR hide everything until a tract is selected and GPkg exists
    if not (selected_tract and gpkg_path and os.path.exists(gpkg_path)):
        return [{'display': 'none'}] * len(ALL_FIELDS)

    tract_id = str(selected_tract)

    # Load the per-county GPKG and get its Region column
    gdf = gpd.read_file(gpkg_path)
    if "Region" not in gdf.columns:
        return [{'display': 'none'}] * len(ALL_FIELDS)

    match = gdf.loc[gdf['id'].astype(str) == tract_id, "Region"]
    if match.empty:
        return [{'display': 'none'}] * len(ALL_FIELDS)

    region = match.iat[0]
    allowed = set(REGION_FIELDS.get(region, []))

    # Build and return the style list based on region
    return [
        {'marginBottom': '10px'} if var in allowed else {'display': 'none'}
        for var, _, _ in ALL_FIELDS
    ]



# --- store_original_prediction ---
@app.callback(
    Output('original_prediction', 'data'),
    [
        Input('selected_census_tract', 'data'),
        Input('model_selector_tab4',  'value')
    ],
    State('editable_gpkg_path', 'data')
)
def store_original_prediction(tract_id, model_file, gpkg_path):
    if not (tract_id and gpkg_path):
        raise PreventUpdate

    # pick suffix + column
    if model_file == "AI2.py":
        suffix, col = "_with_gwr_predictions",  "GWR_Prediction"
    elif model_file == "mgwr_predict.py":
        suffix, col = "_with_mgwr_predictions", "MGWR_Prediction"
    else:
        suffix, col = "_with_predictions",     "Prediction"

    base, ext = os.path.splitext(gpkg_path)
    county_pred = f"{base}{suffix}{ext}"

    # load per‐county if it exists, otherwise global
    if os.path.exists(county_pred):
        gdf = gpd.read_file(county_pred)
    else:
        gdf = gpd.read_file(DEFAULT_PRED_FILES[model_file])
    # if the prediction column itself is missing, bail out
    if col not in gdf.columns:
        return None

    # in store_original_prediction
    subset = gdf.loc[gdf['id'].astype(str) == str(tract_id), col]
    if subset.empty:
        return None
    return subset.iloc[0]

# Single callback to drive the prediction bar for both ForestISO and GWR
@app.callback(
    Output('prediction_bar', 'children'),
    [
        Input('original_prediction',    'data'),
        Input('predictions_refresh',    'data'),
        Input('model_selector_tab4',    'value')
    ],
    [
        State('editable_gpkg_path',     'data'),
        State('selected_census_tract',  'data')
    ]
)
def update_prediction_bar(original_prediction, refresh_val, model_file, gpkg_path, selected_tract):
    # if nothing selected yet
    if not selected_tract or not gpkg_path:
        return "No census tract selected."

    # choose file suffix and column name based on model
    if model_file == "AI2.py":
        suffix, col = "_with_gwr_predictions",  "GWR_Prediction"
    elif model_file == "mgwr_predict.py":
        suffix, col = "_with_mgwr_predictions", "MGWR_Prediction"
    else:
        suffix, col = "_with_predictions",     "Prediction"

    base, ext = os.path.splitext(gpkg_path)
    county_pred_file = f"{base}{suffix}{ext}"

    # load the correct GeoPackage
    if os.path.exists(county_pred_file):
        gdf = gpd.read_file(county_pred_file)
    else:
        # fall back to the global default
        gdf = gpd.read_file(DEFAULT_PRED_FILES[model_file])

    # if column missing
    if col not in gdf.columns:
        return f"Model '{model_file}' has no '{col}' column."

    # grab the current value for this tract
    try:
        current_val = gdf.loc[gdf['id'] == selected_tract, col].iloc[0]
    except Exception:
        return "Selected tract not found."

    # helper for formatting
    def fmt(x):
        try:
            return f"{float(x):.2f}"
        except:
            return str(x)

    return html.Div([
        html.Div(f"Original Prediction: {fmt(original_prediction)}"),
        html.Div(f"Current Prediction:  {fmt(current_val)}"),
    ])

# ----------------------------
# 8. Run the Dash App
# ----------------------------
if __name__ == '__main__':
    # ----------------------------
    # 8.1. Define Data Folder and File Paths
    # ----------------------------
    data_folder = 'data'
    data_final_file = os.path.join(data_folder, 'Data_Final.csv')  # Data_Final.csv path
    census_data_file = os.path.join(data_folder, 'TractData.gpkg')  # GeSoPackage file for Census Data

    # ----------------------------
    # 8.2. Load Data and Define Counties
    # ----------------------------
    data_final_df = load_data_final(data_final_file)
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

    data_by_county = load_all_data_optimized(
        data_final_file, counties
    )

    # Load Census_Tract_data from the GeoPackage file
    census_polygons_by_county = load_census_data(census_data_file)
    logger.debug(f"Census polygons loaded for {len(census_polygons_by_county)} counties.")

    globals()['county_coordinates'] = county_coordinates
    globals()['census_polygons_by_county'] = census_polygons_by_county
    globals()['data_by_county'] = data_by_county

    print("Finished loading webapp.")
    print("127.0.0.1:8050")

    app.run(port="8050", debug=True)

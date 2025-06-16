import dash
from dash import dcc, html, Input, Output, State, callback_context, ctx, clientside_callback, MATCH, ALL
import datetime
import plotly.express as px
import plotly.graph_objects as go  
import pandas as pd
import os
import logging
from flask_caching import Cache
from dash.exceptions import PreventUpdate
from shapely import wkt  
from shapely.geometry import mapping  
import geopandas as gpd 
import math
import json
import subprocess
from chatbot.chatbot_layout import load_chatbot_layout, render_message_bubble
from chatbot.chatbot import generate_response
from analyzer.analyzer import get_insights

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
    dcc.Dropdown(
    id='model_selector_tab4',
    options=[
        {'label': 'ForestISO', 'value': 'AI.py'},
        {'label': 'GWR Model',  'value': 'AI2.py'}
    ],
    value='AI.py',
    clearable=False,
    style={'display': 'none'}    # hidden placeholder
),
    
    html.Div(
    html.Button('Edit Selected Census Tract', id='open_edit_modal', n_clicks=0),
    style={'display': 'none'}
),
    
    html.Div(
    dcc.Dropdown(id='county_selector_tab4', options=[], value=[], multi=True),
    style={'display': 'none'}
),

# Modal for editing county data:
html.Div(
    id='county_edit_modal',
    children=[
        html.H3("Edit County Data"),

        html.Div([
            html.Label("Demographic Index (5-yr ACS)"),
            dcc.Input(id="input_DEMOGIDX_5", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_DEMOGIDX_5", n_clicks=0),
            html.Button("–", id="minus_input_DEMOGIDX_5", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("People of Color (%)"),
            dcc.Input(id="input_PEOPCOLORPCT", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_PEOPCOLORPCT", n_clicks=0),
            html.Button("–", id="minus_input_PEOPCOLORPCT", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Unemployment Rate (%)"),
            dcc.Input(id="input_UNEMPPCT", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_UNEMPPCT", n_clicks=0),
            html.Button("–", id="minus_input_UNEMPPCT", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Residential Land Use (%)"),
            dcc.Input(id="input_pct_residential", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_pct_residential", n_clicks=0),
            html.Button("–", id="minus_input_pct_residential", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Industrial Land Use (%)"),
            dcc.Input(id="input_pct_industrial", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_pct_industrial", n_clicks=0),
            html.Button("–", id="minus_input_pct_industrial", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Retail Land Use (%)"),
            dcc.Input(id="input_pct_retail", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_pct_retail", n_clicks=0),
            html.Button("–", id="minus_input_pct_retail", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Commercial Land Use (%)"),
            dcc.Input(id="input_pct_commercial", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_pct_commercial", n_clicks=0),
            html.Button("–", id="minus_input_pct_commercial", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Annual Average Daily Traffic"),
            dcc.Input(id="input_AADT", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_AADT", n_clicks=0),
            html.Button("–", id="minus_input_AADT", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Avg Commute Distance (Start, mi)"),
            dcc.Input(id="input_Commute_TripMiles_TripStart_avg", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_Commute_TripMiles_TripStart_avg", n_clicks=0),
            html.Button("–", id="minus_input_Commute_TripMiles_TripStart_avg", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Avg Commute Distance (End, mi)"),
            dcc.Input(id="input_Commute_TripMiles_TripEnd_avg", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_Commute_TripMiles_TripEnd_avg", n_clicks=0),
            html.Button("–", id="minus_input_Commute_TripMiles_TripEnd_avg", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Total Population (ACS)"),
            dcc.Input(id="input_ACSTOTPOP", type="number", value=0, step=1),
            html.Button("+", id="plus_input_ACSTOTPOP", n_clicks=0),
            html.Button("–", id="minus_input_ACSTOTPOP", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Demographic Index (2-yr ACS)"),
            dcc.Input(id="input_DEMOGIDX_2", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_DEMOGIDX_2", n_clicks=0),
            html.Button("–", id="minus_input_DEMOGIDX_2", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Poverty Population"),
            dcc.Input(id="input_PovertyPop", type="number", value=0, step=1),
            html.Button("+", id="plus_input_PovertyPop", n_clicks=0),
            html.Button("–", id="minus_input_PovertyPop", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Disability (%)"),
            dcc.Input(id="input_DISABILITYPCT", type="number", value=0, step=0.01),
            html.Button("+", id="plus_input_DISABILITYPCT", n_clicks=0),
            html.Button("–", id="minus_input_DISABILITYPCT", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Biking Trips (Start)"),
            dcc.Input(id="input_BikingTrips(Start)", type="number", value=0, step=1),
            html.Button("+", id="plus_input_BikingTrips(Start)", n_clicks=0),
            html.Button("–", id="minus_input_BikingTrips(Start)", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Biking Trips (End)"),
            dcc.Input(id="input_BikingTrips(End)", type="number", value=0, step=1),
            html.Button("+", id="plus_input_BikingTrips(End)", n_clicks=0),
            html.Button("–", id="minus_input_BikingTrips(End)", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Carpool Trips (Start)"),
            dcc.Input(id="input_CarpoolTrips(Start)", type="number", value=0, step=1),
            html.Button("+", id="plus_input_CarpoolTrips(Start)", n_clicks=0),
            html.Button("–", id="minus_input_CarpoolTrips(Start)", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Carpool Trips (End)"),
            dcc.Input(id="input_CarpoolTrips(End)", type="number", value=0, step=1),
            html.Button("+", id="plus_input_CarpoolTrips(End)", n_clicks=0),
            html.Button("–", id="minus_input_CarpoolTrips(End)", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Commercial Freight Trips (Start)"),
            dcc.Input(id="input_CommercialFreightTrips(Start)", type="number", value=0, step=1),
            html.Button("+", id="plus_input_CommercialFreightTrips(Start)", n_clicks=0),
            html.Button("–", id="minus_input_CommercialFreightTrips(Start)", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Commercial Freight Trips (End)"),
            dcc.Input(id="input_CommercialFreightTrips(End)", type="number", value=0, step=1),
            html.Button("+", id="plus_input_CommercialFreightTrips(End)", n_clicks=0),
            html.Button("–", id="minus_input_CommercialFreightTrips(End)", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Walking Trips (Start)"),
            dcc.Input(id="input_WalkingTrips(Start)", type="number", value=0, step=1),
            html.Button("+", id="plus_input_WalkingTrips(Start)", n_clicks=0),
            html.Button("–", id="minus_input_WalkingTrips(Start)", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Walking Trips (End)"),
            dcc.Input(id="input_WalkingTrips(End)", type="number", value=0, step=1),
            html.Button("+", id="plus_input_WalkingTrips(End)", n_clicks=0),
            html.Button("–", id="minus_input_WalkingTrips(End)", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Public Transit Trips (Start)"),
            dcc.Input(id="input_PublicTransitTrips(Start)", type="number", value=0, step=1),
            html.Button("+", id="plus_input_PublicTransitTrips(Start)", n_clicks=0),
            html.Button("–", id="minus_input_PublicTransitTrips(Start)", n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Div([
            html.Label("Public Transit Trips (End)"),
            dcc.Input(id="input_PublicTransitTrips(End)", type="number", value=0, step=1),
            html.Button("+", id="plus_input_PublicTransitTrips(End)", n_clicks=0),
            html.Button("–", id="minus_input_PublicTransitTrips(End)", n_clicks=0),
        ], style={'marginBottom':'10px'}),


        # ───────── APPLY / RESET / CLOSE ─────────
        html.Div([
            html.Button("Apply Updated Data", id="apply_updated_data", n_clicks=0),
            html.Button("Reset Predictions",    id="reset_predictions",  n_clicks=0),
        ], style={'marginBottom':'10px'}),

        html.Button("Close", id="close_modal", n_clicks=0)
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
                # Right-side: Display AI generated insights
                html.Div(
                    className= 'insights-wrapper',
                    children=[
                        html.Div(
                            id='insight-display-container',
                            children=[
                                html.H1("AI Powered Insights"),
                                dcc.Markdown(
                                    id='insight-content',
                                    children="""
                                    """,
                                )
                            ],
                        className='insight-display-wrapper')])
                
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
                        {'label': 'ForestISO',     'value': 'AI.py'},
                        {'label': 'GWR (local)',  'value': 'AI2.py'},
                        {'label': 'MGWR Model',   'value': 'mgwr_predict.py'},
                    ],
                    value='AI.py',
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
                html.H3("Edit Census Tract Data", style={'fontSize': '14px', 'marginBottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Label("Demographic Index", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(id="input_DEMOGIDX_5", type="number", value=0, step=0.01,
                                style={'width': '80px', 'fontSize': '12px'}),
                        html.Button("+", id="plus_input_DEMOGIDX_5", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'}),
                        html.Button("–", id="minus_input_DEMOGIDX_5", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
                    html.Div([
                        html.Label("People of Color (%)", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(id="input_PEOPCOLORPCT", type="number", value=0, step=0.01,
                                style={'width': '80px', 'fontSize': '12px'}),
                        html.Button("+", id="plus_input_PEOPCOLORPCT", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'}),
                        html.Button("–", id="minus_input_PEOPCOLORPCT", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
                    html.Div([
                        html.Label("Unemployment Rate (%)", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(id="input_UNEMPPCT", type="number", value=0, step=0.01,
                                style={'width': '80px', 'fontSize': '12px'}),
                        html.Button("+", id="plus_input_UNEMPPCT", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'}),
                        html.Button("–", id="minus_input_UNEMPPCT", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
                    html.Div([
                        html.Label("Residential Land Use (%)", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(id="input_pct_residential", type="number", value=0, step=0.01,
                                style={'width': '80px', 'fontSize': '12px'}),
                        html.Button("+", id="plus_input_pct_residential", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'}),
                        html.Button("–", id="minus_input_pct_residential", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
                    html.Div([
                        html.Label("Industrial Land Use (%)", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(id="input_pct_industrial", type="number", value=0, step=0.01,
                                style={'width': '80px', 'fontSize': '12px'}),
                        html.Button("+", id="plus_input_pct_industrial", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'}),
                        html.Button("–", id="minus_input_pct_industrial", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
                    html.Div([
                        html.Label("Retail Land Use (%)", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(id="input_pct_retail", type="number", value=0, step=0.01,
                                style={'width': '80px', 'fontSize': '12px'}),
                        html.Button("+", id="plus_input_pct_retail", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'}),
                        html.Button("–", id="minus_input_pct_retail", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
                    html.Div([
                        html.Label("Commercial Land Use (%)", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(id="input_pct_commercial", type="number", value=0, step=0.01,
                                style={'width': '80px', 'fontSize': '12px'}),
                        html.Button("+", id="plus_input_pct_commercial", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'}),
                        html.Button("–", id="minus_input_pct_commercial", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
                    html.Div([
                        html.Label("Annual Average Daily Traffic", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(id="input_AADT", type="number", value=0, step=0.01,
                                style={'width': '80px', 'fontSize': '12px'}),
                        html.Button("+", id="plus_input_AADT", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'}),
                        html.Button("–", id="minus_input_AADT", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
                    html.Div([
                        html.Label("Avg Commute Distance (Trip Start, mi)", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(id="input_Commute_TripMiles_TripStart_avg", type="number", value=0, step=0.01,
                                style={'width': '80px', 'fontSize': '12px'}),
                        html.Button("+", id="plus_input_Commute_TripMiles_TripStart_avg", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'}),
                        html.Button("–", id="minus_input_Commute_TripMiles_TripStart_avg", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
                    html.Div([
                        html.Label("Avg Commute Distance (Trip End, mi)", style={'fontSize': '12px', 'marginRight': '5px'}),
                        dcc.Input(id="input_Commute_TripMiles_TripEnd_avg", type="number", value=0, step=0.01,
                                style={'width': '80px', 'fontSize': '12px'}),
                        html.Button("+", id="plus_input_Commute_TripMiles_TripEnd_avg", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'}),
                        html.Button("–", id="minus_input_Commute_TripMiles_TripEnd_avg", n_clicks=0,
                                    style={'marginLeft': '5px', 'fontSize': '12px'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'})
                ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '5px'}),
                    html.Div([
                        html.Label("Total Population (ACS)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_ACSTOTPOP", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_ACSTOTPOP", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_ACSTOTPOP", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Demographic Index (2-yr ACS)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_DEMOGIDX_2", type="number", value=0, step=0.01, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_DEMOGIDX_2", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_DEMOGIDX_2", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Poverty Population", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_PovertyPop", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_PovertyPop", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_PovertyPop", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Disability (%)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_DISABILITYPCT", type="number", value=0, step=0.01, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_DISABILITYPCT", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_DISABILITYPCT", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Biking Trips (Start)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_BikingTrips(Start)", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_BikingTrips(Start)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_BikingTrips(Start)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Biking Trips (End)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_BikingTrips(End)", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_BikingTrips(End)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_BikingTrips(End)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Carpool Trips (Start)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_CarpoolTrips(Start)", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_CarpoolTrips(Start)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_CarpoolTrips(Start)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Carpool Trips (End)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_CarpoolTrips(End)", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_CarpoolTrips(End)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_CarpoolTrips(End)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Commercial Freight Trips (Start)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_CommercialFreightTrips(Start)", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_CommercialFreightTrips(Start)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_CommercialFreightTrips(Start)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Commercial Freight Trips (End)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_CommercialFreightTrips(End)", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_CommercialFreightTrips(End)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_CommercialFreightTrips(End)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Walking Trips (Start)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_WalkingTrips(Start)", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_WalkingTrips(Start)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_WalkingTrips(Start)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Walking Trips (End)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_WalkingTrips(End)", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_WalkingTrips(End)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_WalkingTrips(End)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Public Transit Trips (Start)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_PublicTransitTrips(Start)", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_PublicTransitTrips(Start)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_PublicTransitTrips(Start)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                    html.Div([
                        html.Label("Public Transit Trips (End)", style={'fontSize':'12px','marginRight':'5px'}),
                        dcc.Input(id="input_PublicTransitTrips(End)", type="number", value=0, step=1, style={'width':'80px','fontSize':'12px'}),
                        html.Button("+", id="plus_input_PublicTransitTrips(End)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                        html.Button("–", id="minus_input_PublicTransitTrips(End)", n_clicks=0, style={'marginLeft':'5px','fontSize':'12px'}),
                    ], style={'display':'flex','alignItems':'center','marginBottom':'5px'}),

                html.Div([
                    html.Button("Apply Updated Data", id="apply_updated_data", n_clicks=0,
                                style={'fontSize': '12px', 'marginRight': '5px'}),
                    html.Button("Reset Predictions", id="reset_predictions", n_clicks=0,
                                style={'fontSize': '12px'})
                ], style={'marginTop': '10px', 'display': 'flex', 'justifyContent': 'center'})
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
        return load_chatbot_layout(
            [{"sender": "bot", "message": "Hello! I am an interactive safety chatbot designed to provide you with real-time, data-driven insights on roadway safety. Whether you seek information about high-risk areas, traffic incident trends, or general road safety guidance, I will offer reliable and context-aware responses.\n\n" \
            "**Example Prompts**\n\n" \
            "- Can you summarize the crash data from 2020, focusing on common causes?\n\n" \
            "- Show me the top 5 cities with the highest number of crashes in 2021, along with their count.\n\n" \
            }])

@app.callback(
    Output('chat-history-store', 'data'),
    Output('chat-history-container', 'children'),
    Input('clear-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_chat_history(n_clicks):
    if n_clicks and n_clicks > 0:
        # Reset the chat history to an empty list
        # while len(chat_history) > 1:
        #     chat_history.pop()
        return [], [] # Empty list for store, empty list for children
    return dash.no_update, dash.no_update # If button not clicked, do nothing

# --- Python Callback 1: Handle User Input and Display Immediately (with loading placeholder) ---
@app.callback(
    Output('user-input', 'value'),
    Output('chat-history-store', 'data', allow_duplicate=True),
    Output('scroll-trigger', 'data', allow_duplicate=True),
    Output('user-question-for-bot', 'data'),
    Input('send-button', 'n_clicks'),
    State('user-input', 'value'),
    State('chat-history-store', 'data'),
    State('scroll-trigger', 'data'),
    prevent_initial_call=True
)
def handle_user_input(send_button_clicks, user_question, current_chat_data, current_scroll_trigger):
    if not user_question or user_question.strip() == "":
        raise dash.exceptions.PreventUpdate

    # Append user message
    msg = {"sender": "user", "message": user_question}
    current_chat_data.append(msg)
    #chat_history.append(msg)

    # Append temporary loading message
    loading_msg = {"sender": "bot", "message": "Thinking..."}
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

    bot_response_message_content = generate_response(user_question)
    
    # Remove loading message
    current_chat_data.pop()

    msg = {"sender": "bot", "message": bot_response_message_content}
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

    rendered_history_elements = [render_message_bubble(msg['sender'], msg['message']) for msg in stored_chat_data]
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
    Input('predictions_refresh',     'data'),
    State('model_selector_tab4',     'value'),
    State('county_selector_tab4',    'value'),
    State('editable_gpkg_path',      'data'),
)
def update_comparison_graph(refresh, model_file, selected_counties, editable_gpkg_path):
    import os
    # choose GPKG & suffix exactly as in update_predictions_map
    if model_file == "AI2.py":
        suffix, default_file = "_with_gwr_predictions", DEFAULT_PRED_FILES['AI2.py']
    elif model_file == "mgwr_predict.py":
        suffix, default_file = "_with_mgwr_predictions", DEFAULT_PRED_FILES['mgwr_predict.py']
    else:
        suffix, default_file = "_with_predictions",     DEFAULT_PRED_FILES['AI.py']

    if editable_gpkg_path:
        base, ext = os.path.splitext(editable_gpkg_path)
        candidate = f"{base}{suffix}{ext}"
        gpkg_file = candidate if os.path.exists(candidate) else default_file
    else:
        gpkg_file = default_file

    # load it
    gdf = gpd.read_file(gpkg_file)

    # only normalize & filter by CNTY_NAME if that column actually exists
    if selected_counties:
        # First try the AI/GWR style
        if 'CNTY_NAME' in gdf.columns:
            gdf['CNTY_NAME'] = (
                gdf['CNTY_NAME']
                .str.replace(" County", "", regex=False)
                .str.strip()
                .str.title()
            )
            gdf = gdf[gdf['CNTY_NAME'].isin(selected_counties)]

        # Fallback to the MGWR style
        elif 'CountyName' in gdf.columns:
            # Standardize casing just in case
            gdf['CountyName'] = gdf['CountyName'].str.strip().str.title()
            gdf = gdf[gdf['CountyName'].isin(selected_counties)]

    # ensure Prediction column exists
    if 'Prediction' not in gdf.columns:
        return go.Figure()

    # build scatter comparison
    fig = go.Figure()
    # Prediction vs AADT Crash Rate
    # AADT Crash Rate vs Prediction
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

    # VRU Crash Rate vs Prediction
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
        fig = px.scatter_mapbox(
            pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
            lat='Latitude', lon='Longitude', zoom=10, mapbox_style="open-street-map"
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
        fig = px.scatter_mapbox(
            pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
            lat='Latitude', lon='Longitude', zoom=10, mapbox_style="open-street-map"
        )
        fig.update_traces(marker=dict(opacity=0))
    else:
        fig = px.scatter_mapbox(
            df_to_plot,
            lat='Y_Coord', lon='X_Coord', zoom=10, mapbox_style="open-street-map",
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
    Output('insight-content', 'children'),
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
        fig = px.density_mapbox(
            pd.DataFrame(default_center, index=[0]),
            lat='lat', lon='lon',
            radius=radius_px,
            center=default_center, zoom=zoom,
            mapbox_style="open-street-map", opacity=0.7
        )
        fig.update_layout(uirevision=key)
        return fig, "Click 'Apply Filters' to see insights."

    # LOAD YOUR DATA
    df = get_county_data(counties_selected)
    if df.empty:
        # same blank‐dot fallback
        radius_px = convert_miles_to_pixels(radius_miles, zoom, default_center['lat'])
        fig = px.density_mapbox(
            pd.DataFrame(default_center, index=[0]),
            lat='lat', lon='lon',
            radius=radius_px,
            center=default_center, zoom=zoom,
            mapbox_style="open-street-map", opacity=0.7
        )
        fig.update_layout(uirevision=key)
        return fig, "No data found for selected counties. Please adjust filters."

    # APPLY *exactly* the same filters as Tab1, *including* VRU sub‐type
    filtered = filter_data_tab1(
        df,
        start_date, end_date, time_range,
        days_of_week, weather, light, road_surface,
        severity_category, crash_type,
        main_data_type, vru_data_type, 
    )

    filters = {
        "Heatmap Radius (0.1mi - 10mi)": radius_miles,
        "Counties Selected": counties_selected,
        "Main Data Type (All or VRU)": main_data_type,
        "VRU Data Type (All, Bicycle, Pedestrian)": vru_data_type,
        "Start Date": start_date,
        "End Data": end_date,
        "Time Range (12am - 11pm)": time_range,
        "Days of the Week": days_of_week,
        "Weather Condition": weather,
        "Road Surface Condition": road_surface,
        "Light Condition": light,
        "Crash Type": crash_type,
        "Severity Category": severity_category,
    }

    insights = get_insights(filters=filters, filtered_data=filtered, original_data=df)

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
    fig = px.density_mapbox(
        filtered,
        lat='Y_Coord', lon='X_Coord',
        radius=radius_px,
        center={'lat': center_lat, 'lon': center_lon},
        zoom=zoom,
        mapbox_style="open-street-map",
        opacity=0.7,
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
    return fig, insights



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

    # decide which file to load
    if editable_gpkg_path:
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


@app.callback(
    # Existing fields
    Output('input_DEMOGIDX_5', 'value'),
    Output('input_PEOPCOLORPCT', 'value'),
    Output('input_UNEMPPCT', 'value'),
    Output('input_pct_residential', 'value'),
    Output('input_pct_industrial', 'value'),
    Output('input_pct_retail', 'value'),
    Output('input_pct_commercial', 'value'),
    Output('input_AADT', 'value'),
    Output('input_Commute_TripMiles_TripStart_avg', 'value'),
    Output('input_Commute_TripMiles_TripEnd_avg', 'value'),
    Output('input_ACSTOTPOP', 'value'),
    Output('input_DEMOGIDX_2', 'value'),
    Output('input_PovertyPop', 'value'),
    Output('input_DISABILITYPCT', 'value'),
    Output('input_BikingTrips(Start)', 'value'),
    Output('input_BikingTrips(End)', 'value'),
    Output('input_CarpoolTrips(Start)', 'value'),
    Output('input_CarpoolTrips(End)', 'value'),
    Output('input_CommercialFreightTrips(Start)', 'value'),
    Output('input_CommercialFreightTrips(End)', 'value'),
    Output('input_WalkingTrips(Start)', 'value'),
    Output('input_WalkingTrips(End)', 'value'),
    Output('input_PublicTransitTrips(Start)', 'value'),
    Output('input_PublicTransitTrips(End)', 'value'),

    # Inputs: selection + all plus/minus buttons
    Input('selected_census_tract', 'data'),
    Input('plus_input_DEMOGIDX_5', 'n_clicks'),
    Input('minus_input_DEMOGIDX_5', 'n_clicks'),
    Input('plus_input_PEOPCOLORPCT', 'n_clicks'),
    Input('minus_input_PEOPCOLORPCT', 'n_clicks'),
    Input('plus_input_UNEMPPCT', 'n_clicks'),
    Input('minus_input_UNEMPPCT', 'n_clicks'),
    Input('plus_input_pct_residential', 'n_clicks'),
    Input('minus_input_pct_residential', 'n_clicks'),
    Input('plus_input_pct_industrial', 'n_clicks'),
    Input('minus_input_pct_industrial', 'n_clicks'),
    Input('plus_input_pct_retail', 'n_clicks'),
    Input('minus_input_pct_retail', 'n_clicks'),
    Input('plus_input_pct_commercial', 'n_clicks'),
    Input('minus_input_pct_commercial', 'n_clicks'),
    Input('plus_input_AADT', 'n_clicks'),
    Input('minus_input_AADT', 'n_clicks'),
    Input('plus_input_Commute_TripMiles_TripStart_avg', 'n_clicks'),
    Input('minus_input_Commute_TripMiles_TripStart_avg', 'n_clicks'),
    Input('plus_input_Commute_TripMiles_TripEnd_avg', 'n_clicks'),
    Input('minus_input_Commute_TripMiles_TripEnd_avg', 'n_clicks'),
    Input('plus_input_ACSTOTPOP', 'n_clicks'),
    Input('minus_input_ACSTOTPOP', 'n_clicks'),
    Input('plus_input_DEMOGIDX_2', 'n_clicks'),
    Input('minus_input_DEMOGIDX_2', 'n_clicks'),
    Input('plus_input_PovertyPop', 'n_clicks'),
    Input('minus_input_PovertyPop', 'n_clicks'),
    Input('plus_input_DISABILITYPCT', 'n_clicks'),
    Input('minus_input_DISABILITYPCT', 'n_clicks'),
    Input('plus_input_BikingTrips(Start)', 'n_clicks'),
    Input('minus_input_BikingTrips(Start)', 'n_clicks'),
    Input('plus_input_BikingTrips(End)', 'n_clicks'),
    Input('minus_input_BikingTrips(End)', 'n_clicks'),
    Input('plus_input_CarpoolTrips(Start)', 'n_clicks'),
    Input('minus_input_CarpoolTrips(Start)', 'n_clicks'),
    Input('plus_input_CarpoolTrips(End)', 'n_clicks'),
    Input('minus_input_CarpoolTrips(End)', 'n_clicks'),
    Input('plus_input_CommercialFreightTrips(Start)', 'n_clicks'),
    Input('minus_input_CommercialFreightTrips(Start)', 'n_clicks'),
    Input('plus_input_CommercialFreightTrips(End)', 'n_clicks'),
    Input('minus_input_CommercialFreightTrips(End)', 'n_clicks'),
    Input('plus_input_WalkingTrips(Start)', 'n_clicks'),
    Input('minus_input_WalkingTrips(Start)', 'n_clicks'),
    Input('plus_input_WalkingTrips(End)', 'n_clicks'),
    Input('minus_input_WalkingTrips(End)', 'n_clicks'),
    Input('plus_input_PublicTransitTrips(Start)', 'n_clicks'),
    Input('minus_input_PublicTransitTrips(Start)', 'n_clicks'),
    Input('plus_input_PublicTransitTrips(End)', 'n_clicks'),
    Input('minus_input_PublicTransitTrips(End)', 'n_clicks'),

    # States: gpkg path and current values
    State('editable_gpkg_path', 'data'),
    State('input_DEMOGIDX_5', 'value'),
    State('input_PEOPCOLORPCT', 'value'),
    State('input_UNEMPPCT', 'value'),
    State('input_pct_residential', 'value'),
    State('input_pct_industrial', 'value'),
    State('input_pct_retail', 'value'),
    State('input_pct_commercial', 'value'),
    State('input_AADT', 'value'),
    State('input_Commute_TripMiles_TripStart_avg', 'value'),
    State('input_Commute_TripMiles_TripEnd_avg', 'value'),
    State('input_ACSTOTPOP', 'value'),
    State('input_DEMOGIDX_2', 'value'),
    State('input_PovertyPop', 'value'),
    State('input_DISABILITYPCT', 'value'),
    State('input_BikingTrips(Start)', 'value'),
    State('input_BikingTrips(End)', 'value'),
    State('input_CarpoolTrips(Start)', 'value'),
    State('input_CarpoolTrips(End)', 'value'),
    State('input_CommercialFreightTrips(Start)', 'value'),
    State('input_CommercialFreightTrips(End)', 'value'),
    State('input_WalkingTrips(Start)', 'value'),
    State('input_WalkingTrips(End)', 'value'),
    State('input_PublicTransitTrips(Start)', 'value'),
    State('input_PublicTransitTrips(End)', 'value'),
)
def update_modal_values(
    selected_tract,
    plus_demog, minus_demog,
    plus_peop, minus_peop,
    plus_unemp, minus_unemp,
    plus_res, minus_res,
    plus_ind, minus_ind,
    plus_retail, minus_retail,
    plus_comm, minus_comm,
    plus_aadt, minus_aadt,
    plus_c_start, minus_c_start,
    plus_c_end, minus_c_end,
    plus_acstot, minus_acstot,
    plus_demog2, minus_demog2,
    plus_pov, minus_pov,
    plus_dis, minus_dis,
    plus_bike_s, minus_bike_s,
    plus_bike_e, minus_bike_e,
    plus_carp_s, minus_carp_s,
    plus_carp_e, minus_carp_e,
    plus_fre_s, minus_fre_s,
    plus_fre_e, minus_fre_e,
    plus_walk_s, minus_walk_s,
    plus_walk_e, minus_walk_e,
    plus_pt_s, minus_pt_s,
    plus_pt_e, minus_pt_e,
    gpkg_path,
    cur_demog, cur_peop, cur_unemp, cur_res, cur_ind, cur_retail, cur_comm,
    cur_aadt, cur_c_start, cur_c_end,
    cur_acstot, cur_demog2, cur_pov, cur_dis,
    cur_bike_s, cur_bike_e,
    cur_carp_s, cur_carp_e,
    cur_fre_s, cur_fre_e,
    cur_walk_s, cur_walk_e,
    cur_pt_s, cur_pt_e
):
    def clean(v):
        return None if pd.isna(v) else round(v, 2)

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]['prop_id']

    # On new tract selection, load all values
    if trig.startswith('selected_census_tract'):
        if not selected_tract or not gpkg_path:
            raise PreventUpdate
        gdf = gpd.read_file(gpkg_path)
        row = gdf[gdf['id'] == selected_tract]
        if row.empty:
            raise PreventUpdate
        row = row.iloc[0]
        return (
            clean(row.get('DEMOGIDX_5')),
            clean(row.get('PEOPCOLORPCT')),
            clean(row.get('UNEMPPCT')),
            clean(row.get('pct_residential')),
            clean(row.get('pct_industrial')),
            clean(row.get('pct_retail')),
            clean(row.get('pct_commercial')),
            clean(row.get('AADT')),
            clean(row.get('AvgCommuteMiles(Start)')),
            clean(row.get('AvgCommuteMiles(End)')),
            clean(row.get('ACSTOTPOP')),
            clean(row.get('DEMOGIDX_2')),
            clean(row.get('PovertyPop')),
            clean(row.get('DISABILITYPCT')),
            clean(row.get('BikingTrips(Start)')),
            clean(row.get('BikingTrips(End)')),
            clean(row.get('CarpoolTrips(Start)')),
            clean(row.get('CarpoolTrips(End)')),
            clean(row.get('CommercialFreightTrips(Start)')),
            clean(row.get('CommercialFreightTrips(End)')),
            clean(row.get('WalkingTrips(Start)')),
            clean(row.get('WalkingTrips(End)')),
            clean(row.get('PublicTransitTrips(Start)')),
            clean(row.get('PublicTransitTrips(End)')),
        )

    # Otherwise adjust plus/minus
    new_demog      = cur_demog      or 0
    new_peop       = cur_peop       or 0
    new_unemp      = cur_unemp      or 0
    new_res        = cur_res        or 0
    new_ind        = cur_ind        or 0
    new_retail     = cur_retail     or 0
    new_comm       = cur_comm       or 0
    new_aadt       = cur_aadt       or 0
    new_c_start    = cur_c_start    or 0
    new_c_end      = cur_c_end      or 0
    new_acstot     = cur_acstot     or 0
    new_demog2     = cur_demog2     or 0
    new_pov        = cur_pov        or 0
    new_dis        = cur_dis        or 0
    new_bike_s     = cur_bike_s     or 0
    new_bike_e     = cur_bike_e     or 0
    new_carp_s     = cur_carp_s     or 0
    new_carp_e     = cur_carp_e     or 0
    new_fre_s      = cur_fre_s      or 0
    new_fre_e      = cur_fre_e      or 0
    new_walk_s     = cur_walk_s     or 0
    new_walk_e     = cur_walk_e     or 0
    new_pt_s       = cur_pt_s       or 0
    new_pt_e       = cur_pt_e       or 0

    if trig.startswith('plus_input_DEMOGIDX_5'):
        new_demog += 0.1
    elif trig.startswith('minus_input_DEMOGIDX_5'):
        new_demog -= 0.1
    elif trig.startswith('plus_input_PEOPCOLORPCT'):
        new_peop += 0.1
    elif trig.startswith('minus_input_PEOPCOLORPCT'):
        new_peop -= 0.1
    elif trig.startswith('plus_input_UNEMPPCT'):
        new_unemp += 0.1
    elif trig.startswith('minus_input_UNEMPPCT'):
        new_unemp -= 0.1
    elif trig.startswith('plus_input_pct_residential'):
        new_res += 0.1
    elif trig.startswith('minus_input_pct_residential'):
        new_res -= 0.1
    elif trig.startswith('plus_input_pct_industrial'):
        new_ind += 0.1
    elif trig.startswith('minus_input_pct_industrial'):
        new_ind -= 0.1
    elif trig.startswith('plus_input_pct_retail'):
        new_retail += 0.1
    elif trig.startswith('minus_input_pct_retail'):
        new_retail -= 0.1
    elif trig.startswith('plus_input_pct_commercial'):
        new_comm += 0.1
    elif trig.startswith('minus_input_pct_commercial'):
        new_comm -= 0.1
    elif trig.startswith('plus_input_AADT'):
        new_aadt += 1
    elif trig.startswith('minus_input_AADT'):
        new_aadt -= 1
    elif trig.startswith('plus_input_Commute_TripMiles_TripStart_avg'):
        new_c_start += 0.1
    elif trig.startswith('minus_input_Commute_TripMiles_TripStart_avg'):
        new_c_start -= 0.1
    elif trig.startswith('plus_input_Commute_TripMiles_TripEnd_avg'):
        new_c_end += 0.1
    elif trig.startswith('minus_input_Commute_TripMiles_TripEnd_avg'):
        new_c_end -= 0.1
    elif trig.startswith('plus_input_ACSTOTPOP'):
        new_acstot += 1
    elif trig.startswith('minus_input_ACSTOTPOP'):
        new_acstot -= 1
    elif trig.startswith('plus_input_DEMOGIDX_2'):
        new_demog2 += 0.1
    elif trig.startswith('minus_input_DEMOGIDX_2'):
        new_demog2 -= 0.1
    elif trig.startswith('plus_input_PovertyPop'):
        new_pov += 1
    elif trig.startswith('minus_input_PovertyPop'):
        new_pov -= 1
    elif trig.startswith('plus_input_DISABILITYPCT'):
        new_dis += 0.1
    elif trig.startswith('minus_input_DISABILITYPCT'):
        new_dis -= 0.1
    elif trig.startswith('plus_input_BikingTrips(Start)'):
        new_bike_s += 1
    elif trig.startswith('minus_input_BikingTrips(Start)'):
        new_bike_s -= 1
    elif trig.startswith('plus_input_BikingTrips(End)'):
        new_bike_e += 1
    elif trig.startswith('minus_input_BikingTrips(End)'):
        new_bike_e -= 1
    elif trig.startswith('plus_input_CarpoolTrips(Start)'):
        new_carp_s += 1
    elif trig.startswith('minus_input_CarpoolTrips(Start)'):
        new_carp_s -= 1
    elif trig.startswith('plus_input_CarpoolTrips(End)'):
        new_carp_e += 1
    elif trig.startswith('minus_input_CarpoolTrips(End)'):
        new_carp_e -= 1
    elif trig.startswith('plus_input_CommercialFreightTrips(Start)'):
        new_fre_s += 1
    elif trig.startswith('minus_input_CommercialFreightTrips(Start)'):
        new_fre_s -= 1
    elif trig.startswith('plus_input_CommercialFreightTrips(End)'):
        new_fre_e += 1
    elif trig.startswith('minus_input_CommercialFreightTrips(End)'):
        new_fre_e -= 1
    elif trig.startswith('plus_input_WalkingTrips(Start)'):
        new_walk_s += 1
    elif trig.startswith('minus_input_WalkingTrips(Start)'):
        new_walk_s -= 1
    elif trig.startswith('plus_input_WalkingTrips(End)'):
        new_walk_e += 1
    elif trig.startswith('minus_input_WalkingTrips(End)'):
        new_walk_e -= 1
    elif trig.startswith('plus_input_PublicTransitTrips(Start)'):
        new_pt_s += 1
    elif trig.startswith('minus_input_PublicTransitTrips(Start)'):
        new_pt_s -= 1
    elif trig.startswith('plus_input_PublicTransitTrips(End)'):
        new_pt_e += 1
    elif trig.startswith('minus_input_PublicTransitTrips(End)'):
        new_pt_e -= 1

    return (
        round(new_demog, 2),
        round(new_peop, 2),
        round(new_unemp, 2),
        round(new_res, 2),
        round(new_ind, 2),
        round(new_retail, 2),
        round(new_comm, 2),
        round(new_aadt, 2),
        round(new_c_start, 2),
        round(new_c_end, 2),
        round(new_acstot, 2),
        round(new_demog2, 2),
        round(new_pov, 2),
        round(new_dis, 2),
        round(new_bike_s, 2),
        round(new_bike_e, 2),
        round(new_carp_s, 2),
        round(new_carp_e, 2),
        round(new_fre_s, 2),
        round(new_fre_e, 2),
        round(new_walk_s, 2),
        round(new_walk_e, 2),
        round(new_pt_s, 2),
        round(new_pt_e, 2),
    )

    
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
    Output('selected_census_tract', 'data'),
    Input('predictions_map', 'clickData')
)
def store_selected_tract(clickData):
    if clickData and 'points' in clickData:
        tract_id = clickData['points'][0].get('location')
        return tract_id
    raise PreventUpdate


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
    Output('predictions_refresh', 'data', allow_duplicate=True),
    Input('reset_predictions', 'n_clicks'),
    State('editable_gpkg_path', 'data'),
    State('predictions_refresh', 'data'),
    prevent_initial_call=True
)
def reset_predictions(n_clicks, editable_gpkg_path, current_refresh):
    if n_clicks:
        if editable_gpkg_path and os.path.exists(editable_gpkg_path):
            base, ext = os.path.splitext(editable_gpkg_path)
            predictions_file = base + "_with_predictions" + ext
            if os.path.exists(predictions_file):
                try:
                    os.remove(predictions_file)
                    logger.info(f"Deleted predictions file: {predictions_file}")
                except Exception as e:
                    logger.error(f"Error deleting predictions file {predictions_file}: {e}")
            else:
                logger.info("No editable predictions file to delete.")
        else:
            logger.info("No editable GPkg provided; nothing to reset.")
        # Increment the refresh value to trigger a refresh
        return (current_refresh or 0) + 1
    raise PreventUpdate



# Callback to handle county‐selection, data‐editing, apply/reset predictions
@app.callback(
    Output('editable_gpkg_path', 'data'),
    Output('predictions_refresh', 'data'),
    [
        Input('county_selector_tab4',     'value'),
        Input('apply_updated_data',       'n_clicks'),
        Input('reset_predictions',        'n_clicks'),
        Input('model_selector_tab4',      'value'),
    ],
    [
        State('selected_census_tract',                   'data'),
        State('editable_gpkg_path',                      'data'),
        # existing fields
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
        # ───────── NEW STATES ─────────
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
    ]
)
def update_editable_gpkg_and_predictions(
    county_selector_value,
    apply_n_clicks,
    reset_n_clicks,
    model_file,
    selected_tract,
    gpkg_path,
    demogidx_5,
    peopcolorpct,
    unemppct,
    pct_residential,
    pct_industrial,
    pct_retail,
    pct_commercial,
    aadt,
    commute_start,
    commute_end,
    acstotpop,
    demogidx_2,
    poverty_pop,
    disabilitypct,
    biking_start,
    biking_end,
    carpool_start,
    carpool_end,
    freight_start,
    freight_end,
    walking_start,
    walking_end,
    transit_start,
    transit_end,
    current_refresh
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered = ctx.triggered[0]['prop_id'].split('.')[0]

    # 1) New county selected → create an editable GPKG
    if triggered == 'county_selector_tab4':
        if county_selector_value and len(county_selector_value) == 1:
            county = county_selector_value[0]

            # Pick source gpkg & dest folder per model
            if model_file == 'AI2.py':
                source = DEFAULT_PRED_FILES['AI2.py']
                dest = './AI/'
            elif model_file == 'mgwr_predict.py':
                source = DEFAULT_PRED_FILES['mgwr_predict.py']
                dest = './MGWR/'
            else:  # AI.py
                source = DEFAULT_PRED_FILES['AI.py']
                dest = './AI/'

            # copy that county out of the right gpkg
            return copy_county_gpkg(county, source_gpkg=source, dest_folder=dest), current_refresh
        else:
            raise PreventUpdate


    # 2) Apply updates to the tract and re‐run the model
    elif triggered == 'apply_updated_data':
        if apply_n_clicks and selected_tract and gpkg_path:
            gdf = gpd.read_file(gpkg_path)
            idx = gdf[gdf['id'] == selected_tract].index
            if idx.empty:
                raise PreventUpdate

            # ─── Existing fields ───
            gdf.loc[idx, 'DEMOGIDX_5']                         = demogidx_5
            gdf.loc[idx, 'PEOPCOLORPCT']                       = peopcolorpct
            gdf.loc[idx, 'UNEMPPCT']                           = unemppct
            gdf.loc[idx, 'pct_residential']                    = pct_residential
            gdf.loc[idx, 'pct_industrial']                     = pct_industrial
            gdf.loc[idx, 'pct_retail']                         = pct_retail
            gdf.loc[idx, 'pct_commercial']                     = pct_commercial
            gdf.loc[idx, 'AADT']                               = aadt
            gdf.loc[idx, 'Commute_TripMiles_TripStart_avg']    = commute_start
            gdf.loc[idx, 'Commute_TripMiles_TripEnd_avg']      = commute_end

            # ─── NEW fields ───
            gdf.loc[idx, 'ACSTOTPOP']                          = acstotpop
            gdf.loc[idx, 'DEMOGIDX_2']                         = demogidx_2
            gdf.loc[idx, 'PovertyPop']                         = poverty_pop
            gdf.loc[idx, 'DISABILITYPCT']                      = disabilitypct
            gdf.loc[idx, 'BikingTrips(Start)']                 = biking_start
            gdf.loc[idx, 'BikingTrips(End)']                   = biking_end
            gdf.loc[idx, 'CarpoolTrips(Start)']                = carpool_start
            gdf.loc[idx, 'CarpoolTrips(End)']                  = carpool_end
            gdf.loc[idx, 'CommercialFreightTrips(Start)']      = freight_start
            gdf.loc[idx, 'CommercialFreightTrips(End)']        = freight_end
            gdf.loc[idx, 'WalkingTrips(Start)']                = walking_start
            gdf.loc[idx, 'WalkingTrips(End)']                  = walking_end
            gdf.loc[idx, 'PublicTransitTrips(Start)']          = transit_start
            gdf.loc[idx, 'PublicTransitTrips(End)']            = transit_end

            # overwrite the editable GPKG
            gdf.to_file(gpkg_path, driver="GPKG")

            # re‐run the selected model script
            import sys
            base, ext = os.path.splitext(gpkg_path)

            # choose suffix per model
            if model_file == 'AI2.py':
                suffix = '_with_gwr_predictions'
            elif model_file == 'mgwr_predict.py':
                suffix = '_with_mgwr_predictions'
            else:  # AI.py
                suffix = '_with_predictions'

            output_file = f"{base}{suffix}{ext}"

            try:
                subprocess.run(
                    [sys.executable, model_file, gpkg_path, output_file],
                    check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError:
                raise PreventUpdate

            return gpkg_path, (current_refresh or 0) + 1
        else:
            raise PreventUpdate

    # 3) Reset: delete editable & predictions files, re‐copy county
    elif triggered == 'reset_predictions':
        if reset_n_clicks:
            if gpkg_path and os.path.exists(gpkg_path):
                os.remove(gpkg_path)
            base, ext = os.path.splitext(gpkg_path or '')
            suffix = '_with_gwr_predictions' if model_file == 'AI2.py' else '_with_predictions'
            pred_file = f"{base}{suffix}{ext}"
            if os.path.exists(pred_file):
                os.remove(pred_file)

            new_path = (
                copy_county_gpkg(county_selector_value[0])
                if county_selector_value and len(county_selector_value)==1 else
                None
            )
            return new_path, (current_refresh or 0) + 1
        else:
            raise PreventUpdate

    else:
        raise PreventUpdate


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

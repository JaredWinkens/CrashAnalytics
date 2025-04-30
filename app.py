import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go  # Added import for graph_objects
import pandas as pd
import os
import logging
from flask_caching import Cache
from dash.exceptions import PreventUpdate
from shapely import wkt  # Added import for shapely WKT parsing
from shapely.geometry import mapping  # Added import for geometry mapping
import geopandas as gpd  # Import GeoPandas for GeoPackage reading
import math
import json
import subprocess

# where your two models dump their “big” prediction files:
DEFAULT_PRED_FILES = {
    'AI.py':  './AI/Large_DataSet2.25_with_predictions.gpkg',
    'AI2.py': './AI/Rename_DataSet2.25_with_gwr_predictions.gpkg'
}

# ----------------------------
# 1. Setup Logging
# ----------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------
# 2. Data Loading and Preprocessing
# ----------------------------


def copy_county_gpkg(county, source_gpkg='./AI/Large_DataSet2.25_with_predictions.gpkg', dest_folder='./AI/'):
    try:
        gdf = gpd.read_file(source_gpkg)
        # Standardize CNTY_NAME by removing the trailing " County"
        gdf['CNTY_NAME'] = gdf['CNTY_NAME'].str.replace(" County", "", regex=False).str.strip().str.title()
        county_gdf = gdf[gdf['CNTY_NAME'] == county]
        if county_gdf.empty:
            logger.error(f"No data found for county: {county}")
            raise PreventUpdate
        # Add an 'id' column for later matching
        county_gdf = county_gdf.copy()
        county_gdf['id'] = county_gdf.index.astype(str)
        dest_file = os.path.join(dest_folder, f"{county}_editable.gpkg")
        county_gdf.to_file(dest_file, driver='GPKG')
        logger.debug(f"Created editable GPkg for {county} at {dest_file}")
        return dest_file
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
    Load and preprocess Data_Final.csv.
    """
    usecols = [
        'CaseNumber',             # Case_Number
        'CrashDate',              # Crash_Date
        'CrashTimeFormatted',     # Crash_Time
        'RoadSurfaceCondition',   # RoadSurfac (Corrected)
        'WeatherCondition',       # WeatherCon
        'LightCondition',         # LightCon
        'CountyName',             # County
        'CrashCategory',          # Data_Type
        'CrashType',              # New column for VRU type
        'Longitude',              # X_Coord
        'Latitude'                # Y_Coord
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
        'CrashType': str,         # New dtype
        'Longitude': float,
        'Latitude': float
    }

    chunks = []
    try:
        for i, chunk in enumerate(pd.read_csv(
            file_path,
            usecols=usecols,
            dtype=dtype,
            chunksize=100000,
            header=0
        )):
            logger.debug(f"Data_Final.csv Chunk {i} columns: {chunk.columns.tolist()}")

            # Rename columns to match the main DataFrame structure
            chunk = chunk.rename(columns={
                'CaseNumber': 'Case_Number',
                'CrashDate': 'Crash_Date',
                'CrashTimeFormatted': 'Crash_Time',
                'RoadSurfaceCondition': 'RoadSurfac',  # Corrected renaming
                'WeatherCondition': 'WeatherCon',
                'LightCondition': 'LightCon',
                'CrashCategory': 'Data_Type',
                'CrashType': 'Crash_Type',   # New renaming
                'Longitude': 'X_Coord',
                'Latitude': 'Y_Coord',
                'CountyName': 'County'
            })

            # Verify 'County' column exists after renaming
            if 'County' not in chunk.columns:
                logger.warning(f"'CountyName' column missing in chunk {i}. Assigning 'Unknown'.")
                chunk['County'] = 'Unknown'

            # Standardize 'County' names
            chunk['County'] = chunk['County'].apply(standardize_county_name)

            # Convert date and time columns
            chunk['Crash_Date'] = pd.to_datetime(chunk['Crash_Date'], errors='coerce')
            chunk['Crash_Time'] = pd.to_datetime(chunk['Crash_Time'], format='%I:%M %p', errors='coerce').dt.hour

            # Handle missing values and clean string columns
            chunk['WeatherCon'] = chunk['WeatherCon'].fillna('Unknown').str.strip().str.title()
            chunk['LightCon'] = chunk['LightCon'].fillna('Unknown').str.strip().str.title()
            chunk['RoadSurfac'] = chunk['RoadSurfac'].fillna('Unknown').str.strip().str.title()
            chunk['Data_Type'] = chunk['Data_Type'].fillna('Non-VRU').str.strip().str.upper()
            
            # New: Process Crash_Type column (convert to uppercase)
            chunk['Crash_Type'] = chunk['Crash_Type'].fillna('Unknown').str.strip().str.upper()

            # Add 'Sex' column as 'Unknown' since Data_Final.csv does not have this information
            chunk['Sex'] = 'Unknown'

            # Reorder columns to match the main DataFrame – now including Crash_Type
            chunk = chunk[
                ['Case_Number', 'X_Coord', 'Y_Coord', 'Crash_Date',
                 'Crash_Time', 'Sex', 'WeatherCon', 'LightCon', 'RoadSurfac',
                 'Data_Type', 'County', 'Crash_Type']
            ]

            chunks.append(chunk)
            logger.debug(f"Data_Final.csv Chunk {i} loaded with {len(chunk)} records.")

        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.DataFrame(columns=[
                'Case_Number', 'X_Coord', 'Y_Coord', 'Crash_Date',
                'Crash_Time', 'Sex', 'WeatherCon', 'LightCon', 'RoadSurfac',
                'Data_Type', 'County', 'Crash_Type'
            ])
            logger.warning(f"No data found in {file_path}.")

        logger.debug(f"Total records loaded from Data_Final.csv: {len(df)}")
        return df

    except Exception as e:
        logger.error(f"Error loading Data_Final.csv: {e}")
        return pd.DataFrame(columns=[
            'Case_Number', 'X_Coord', 'Y_Coord', 'Crash_Date',
            'Crash_Time', 'Sex', 'WeatherCon', 'LightCon', 'RoadSurfac',
            'Data_Type', 'County', 'Crash_Type'
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
        'Crash_Time', 'Sex', 'WeatherCon', 'LightCon', 'RoadSurfac', 'Data_Type', 'County'
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
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Crash Data Analytics'
server = app.server

# Initialize caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple'  # For development; use Redis or similar in production
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

def filter_data_tab1(df, start_date, end_date, time_range, days_of_week, gender, weather, light, road_surface, main_data_type, vru_data_type):
    # Date filtering
    if start_date and end_date:
        df = df[(df['Crash_Date'] >= start_date) & (df['Crash_Date'] <= end_date)]
    # Time filtering
    if time_range:
        df = df[(df['Crash_Time'] >= time_range[0]) & (df['Crash_Time'] <= time_range[1])]
    # Day of week filtering
    if days_of_week:
        df = df[df['Crash_Date'].dt.day_name().isin(days_of_week)]
    # Gender, Weather, Light, and Road Surface filtering
    if gender != 'All':
        df = df[df['Sex'] == gender]
    if weather != 'All':
        df = df[df['WeatherCon'] == weather]
    if light != 'All':
        df = df[df['LightCon'] == light]
    if road_surface != 'All':
        df = df[df['RoadSurfac'] == road_surface]
    
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
        pass  # no additional filtering on crash type
    elif main_data_type == 'None':
        # "None" returns no crash data.
        df = df.iloc[0:0]
    
    return df


# ----------------------------
# 5. Define UI Components
# ----------------------------

def common_controls(prefix, show_buttons, available_counties, unique_weather, unique_light, unique_road):
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
        html.Label('Select Gender:', style={'margin-top': '20px'}),
        dcc.RadioItems(
            id=f'gender_selector_{prefix}',
            options=[
                {'label': 'All', 'value': 'All'},
                {'label': 'Male', 'value': 'Male'},
                {'label': 'Female', 'value': 'Female'},
                {'label': 'Other', 'value': 'Other'}
            ],
            value='All',
            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
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
        )
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
    # Grab the exact, normalized county names from your loaded GeoPackage
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
        dcc.Tab(label='Predictions', value='tab-4')
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
    dcc.Download(id='download_data_tab3'),   # For Census Data tab
    
    # (Place these near your other dcc.Store definitions in your app.layout)
    dcc.Store(id='editable_gpkg_path'),
    dcc.Store(id='selected_census_tract'),
    dcc.Store(id='predictions_refresh', data=0),  # New store for refresh trigger
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
            html.Label("Demographic Index (5‑yr ACS)"),
            dcc.Input(id="input_DEMOGIDX_5", type="number", value=None, step=0.01)
        ]),
        html.Div([
            html.Label("People of Color (%)"),
            dcc.Input(id="input_PEOPCOLORPCT", type="number", value=None, step=0.01)
        ]),
        html.Div([
            html.Label("Unemployment Rate (%)"),
            dcc.Input(id="input_UNEMPPCT", type="number", value=None, step=0.01)
        ]),
        html.Div([
            html.Label("Residential Land Use (%)"),
            dcc.Input(id="input_pct_residential", type="number", value=None, step=0.01)
        ]),
        html.Div([
            html.Label("Industrial Land Use (%)"),
            dcc.Input(id="input_pct_industrial", type="number", value=None, step=0.01)
        ]),
        html.Div([
            html.Label("Retail Land Use (%)"),
            dcc.Input(id="input_pct_retail", type="number", value=None, step=0.01)
        ]),
        html.Div([
            html.Label("Commercial Land Use (%)"),
            dcc.Input(id="input_pct_commercial", type="number", value=None, step=0.01)
        ]),
        html.Div([
            html.Label("Annual Average Daily Traffic"),
            dcc.Input(id="input_AADT", type="number", value=None, step=0.01)
        ]),
        html.Div([
            html.Label("Avg Commute Distance (Trip Start, mi)"),
            dcc.Input(id="input_Commute_TripMiles_TripStart_avg", type="number", value=None, step=0.01)
        ]),
        html.Div([
            html.Label("Avg Commute Distance (Trip End, mi)"),
            dcc.Input(id="input_Commute_TripMiles_TripEnd_avg", type="number", value=None, step=0.01)
        ]),
        html.Div([
            html.Button("Apply Updated Data", id="apply_updated_data", n_clicks=0, style={'marginRight': '10px'}),
            html.Button("Reset Predictions", id="reset_predictions", n_clicks=0),
        ], style={'marginTop': '15px'}),
        html.Button("Close", id="close_modal", n_clicks=0, style={'marginTop': '15px'})
    ],
    style={
        'display': 'none',  # Hidden by default
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

    if tab == 'tab-1':
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
                                unique_road=unique_road
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




    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                # Left-side: New slider block *above* the common controls
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
                
                # Now include your common controls (which starts with the county selector)
                common_controls(
                    'tab2',
                    show_buttons=True,
                    available_counties=available_counties,
                    unique_weather=unique_weather,
                    unique_light=unique_light,
                    unique_road=unique_road
                )
            ], className='responsive-controls'),
            
            # Right-side: The Heatmap Graph container
            html.Div(
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
                className='responsive-graph'
            )
        ], className='desktop-layout')


    elif tab == 'tab-3':
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
                # Editing Controls – one row per field; add additional rows as needed.
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
                    html.Button("Apply Updated Data", id="apply_updated_data", n_clicks=0,
                                style={'fontSize': '12px', 'marginRight': '5px'}),
                    html.Button("Reset Predictions", id="reset_predictions", n_clicks=0,
                                style={'fontSize': '12px'})
                ], style={'marginTop': '10px', 'display': 'flex', 'justifyContent': 'center'})
            ], className='responsive-controls'),
            # Right-side: Predictions Map
            html.Div(
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
                    config={'modeBarButtonsToRemove': ['lasso2d'], 'displayModeBar': True, 'scrollZoom': True}
                ),
                className='responsive-graph'
            )
        ], className='desktop-layout')






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
        State('gender_selector_tab1', 'value'),
        State('weather_selector_tab1', 'value'),
        State('light_selector_tab1', 'value'),
        State('road_surface_selector_tab1', 'value'),
        State('data_type_selector_main_tab1', 'value'),
        State('data_type_selector_vru_tab1', 'value')
    ]
)
def map_tab1(apply_n_clicks, clear_n_clicks, counties_selected, selected_data,
             start_date, end_date, time_range, days_of_week,
             gender, weather, light, road_surface, main_data_type, vru_data_type):
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
            days_of_week, gender, weather, light, road_surface,
            main_data_type, vru_data_type
        )
        if selected_data and 'points' in selected_data:
            keep = [pt['customdata'][0] for pt in selected_data['points']]
            filtered = filtered[filtered['Case_Number'].isin(keep)]
        df_to_plot = filtered
        out_selected = selected_data

    elif triggered == 'clear_drawing_tab1':
        # reapply filters but drop box selection
        df_to_plot = filter_data_tab1(
            df, start_date, end_date, time_range,
            days_of_week, gender, weather, light, road_surface,
            main_data_type, vru_data_type
        )
        out_selected = None

    else:  # initial_load
        df_to_plot = filter_data_tab1(
            df, '1900-01-01', '1901-01-01', [0, 23],
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
            'All','All','All','All',
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

    # **persist view** except when counties change
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
        State('gender_selector_tab1', 'value'),
        State('weather_selector_tab1', 'value'),
        State('light_selector_tab1', 'value'),
        State('road_surface_selector_tab1', 'value'),
        State('data_type_selector_main_tab1', 'value'),
        State('data_type_selector_vru_tab1', 'value'),
        State('scatter_map', 'selectedData')
    ]
)
def download_filtered_data_tab1(n_clicks, counties_selected, start_date, end_date, time_range, days_of_week,
                                gender, weather, light, road_surface, main_data_type, vru_data_type, selected_data):
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
                days_of_week, gender, weather, light, road_surface,
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
    [
        Input('apply_filter_tab2', 'n_clicks')
    ],
    [
        State('radius_slider_tab2', 'value'),
        State('county_selector_tab2', 'value'),
        State('data_type_selector_main_tab2', 'value'),
        State('data_type_selector_vru_tab2', 'value'),
        State('date_picker_tab2', 'start_date'),
        State('date_picker_tab2', 'end_date'),
        State('time_slider_tab2', 'value'),
        State('day_of_week_checklist_tab2', 'value'),
        State('gender_selector_tab2', 'value'),
        State('weather_selector_tab2', 'value'),
        State('light_selector_tab2', 'value'),
        State('road_surface_selector_tab2', 'value')
    ]
)
def update_heatmap_tab2(apply_n_clicks, radius_miles, counties_selected, main_data_type, vru_data_type,
                        start_date, end_date, time_range, days_of_week,
                        gender, weather, light, road_surface):
    zoom = 10
    default_center = {'lat': 40.7128, 'lon': -74.0060}
    # use counties as key so changing counties resets view, otherwise persists
    key = 'tab2-' + '-'.join(sorted(counties_selected or []))

    # before first click: show default
    if not apply_n_clicks:
        radius_px = convert_miles_to_pixels(radius_miles, zoom, default_center['lat'])
        fig = px.density_mapbox(
            pd.DataFrame({'Latitude': [default_center['lat']], 'Longitude': [default_center['lon']]}),
            lat='Latitude', lon='Longitude',
            radius=radius_px,
            center=default_center, zoom=zoom,
            mapbox_style="open-street-map", opacity=0.7
        )
        fig.update_layout(uirevision=key)
        return fig

    df = get_county_data(counties_selected)
    if df.empty:
        radius_px = convert_miles_to_pixels(radius_miles, zoom, default_center['lat'])
        fig = px.density_mapbox(
            pd.DataFrame({'Latitude': [default_center['lat']], 'Longitude': [default_center['lon']]}),
            lat='Latitude', lon='Longitude',
            radius=radius_px,
            center=default_center, zoom=zoom,
            mapbox_style="open-street-map", opacity=0.7
        )
        fig.update_layout(uirevision=key)
        return fig

    # apply same filters as tab1
    filtered = filter_data_tab1(
        df, start_date, end_date, time_range,
        days_of_week, gender, weather, light, road_surface,
        main_data_type, vru_data_type
    )

    # determine center
    if 'All' in counties_selected:
        center_lat = df['Y_Coord'].mean()
        center_lon = df['X_Coord'].mean()
    else:
        center_lat = sum(county_coordinates[c]['lat'] for c in counties_selected) / len(counties_selected)
        center_lon = sum(county_coordinates[c]['lon'] for c in counties_selected) / len(counties_selected)

    # convert miles to pixels
    radius_px = convert_miles_to_pixels(radius_miles, zoom, center_lat)

    # build heatmap
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

    # persist camera/zoom except when counties change
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
        suffix, pred_col = "_with_gwr_predictions", "GWR_Prediction"
        default_file = DEFAULT_PRED_FILES['AI2.py']
    else:
        suffix, pred_col = "_with_predictions", "Prediction"
        default_file = DEFAULT_PRED_FILES['AI.py']

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
        if 'CNTY_NAME' in gdf:
            gdf['CNTY_NAME'] = (
                gdf['CNTY_NAME']
                   .str.replace(" County", "", regex=False)
                   .str.strip()
                   .str.title()
            )
        if selected_counties:
            gdf = gdf[gdf['CNTY_NAME'].isin(selected_counties)]

        valid   = gdf[~gdf['Prediction'].isna()]
        missing = gdf[ gdf['Prediction'].isna()]

        fig = go.Figure([
            go.Choroplethmapbox(
                geojson=json.loads(valid.to_json()),
                locations=valid['id'],
                z=valid['Prediction'],
                colorscale='YlGnBu',
                marker_opacity=0.6,
                marker_line_width=1,
                colorbar=dict(title="Prediction"),
                featureidkey="properties.id",
                name="Prediction"
            ),
            go.Choroplethmapbox(
                geojson=json.loads(missing.to_json()),
                locations=missing['id'],
                z=[0]*len(missing),
                colorscale=[[0,"black"],[1,"black"]],
                marker_opacity=0.9,
                marker_line_width=1,
                showscale=False,
                featureidkey="properties.id",
                name="Missing"
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
    # Inputs for modal population and all plus/minus buttons:
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
    # State: editable GPkg and current field values
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
    State('input_Commute_TripMiles_TripEnd_avg', 'value')
)
def update_modal_values(selected_tract,
                        plus_demog, minus_demog,
                        plus_peop, minus_peop,
                        plus_unemp, minus_unemp,
                        plus_res, minus_res,
                        plus_ind, minus_ind,
                        plus_retail, minus_retail,
                        plus_commercial, minus_commercial,
                        plus_aadt, minus_aadt,
                        plus_commute_start, minus_commute_start,
                        plus_commute_end, minus_commute_end,
                        gpkg_path,
                        cur_demog, cur_peop, cur_unemp, cur_res,
                        cur_ind, cur_retail, cur_commercial,
                        cur_aadt, cur_commute_start, cur_commute_end):
    def clean(v):
        return None if pd.isna(v) else round(v, 2)
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_prop = ctx.triggered[0]['prop_id']

    # If the census tract was selected, load new values from the GPkg:
    if triggered_prop.startswith('selected_census_tract'):
        if not selected_tract or not gpkg_path:
            raise PreventUpdate
        try:
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
            )
        except Exception as e:
            logger.error(f"Error populating modal: {e}")
            raise PreventUpdate

    # Otherwise, if a plus/minus button was clicked, update the corresponding field:
    # Start with current values (or 0 if None)
    new_demog = (cur_demog or 0)
    new_peop = (cur_peop or 0)
    new_unemp = (cur_unemp or 0)
    new_res = (cur_res or 0)
    new_ind = (cur_ind or 0)
    new_retail = (cur_retail or 0)
    new_commercial = (cur_commercial or 0)
    new_aadt = (cur_aadt or 0)
    new_commute_start = (cur_commute_start or 0)
    new_commute_end = (cur_commute_end or 0)

    if triggered_prop.startswith('plus_input_DEMOGIDX_5'):
        new_demog += 0.1
    elif triggered_prop.startswith('minus_input_DEMOGIDX_5'):
        new_demog -= 0.1
    elif triggered_prop.startswith('plus_input_PEOPCOLORPCT'):
        new_peop += 0.1
    elif triggered_prop.startswith('minus_input_PEOPCOLORPCT'):
        new_peop -= 0.1
    elif triggered_prop.startswith('plus_input_UNEMPPCT'):
        new_unemp += 0.1
    elif triggered_prop.startswith('minus_input_UNEMPPCT'):
        new_unemp -= 0.1
    elif triggered_prop.startswith('plus_input_pct_residential'):
        new_res += 0.1
    elif triggered_prop.startswith('minus_input_pct_residential'):
        new_res -= 0.1
    elif triggered_prop.startswith('plus_input_pct_industrial'):
        new_ind += 0.1
    elif triggered_prop.startswith('minus_input_pct_industrial'):
        new_ind -= 0.1
    elif triggered_prop.startswith('plus_input_pct_retail'):
        new_retail += 0.1
    elif triggered_prop.startswith('minus_input_pct_retail'):
        new_retail -= 0.1
    elif triggered_prop.startswith('plus_input_pct_commercial'):
        new_commercial += 0.1
    elif triggered_prop.startswith('minus_input_pct_commercial'):
        new_commercial -= 0.1
    elif triggered_prop.startswith('plus_input_AADT'):
        new_aadt += 0.1
    elif triggered_prop.startswith('minus_input_AADT'):
        new_aadt -= 0.1
    elif triggered_prop.startswith('plus_input_Commute_TripMiles_TripStart_avg'):
        new_commute_start += 0.1
    elif triggered_prop.startswith('minus_input_Commute_TripMiles_TripStart_avg'):
        new_commute_start -= 0.1
    elif triggered_prop.startswith('plus_input_Commute_TripMiles_TripEnd_avg'):
        new_commute_end += 0.1
    elif triggered_prop.startswith('minus_input_Commute_TripMiles_TripEnd_avg'):
        new_commute_end -= 0.1

    return (round(new_demog, 2), round(new_peop, 2), round(new_unemp, 2), round(new_res, 2), round(new_ind, 2),
            round(new_retail, 2), round(new_commercial, 2), round(new_aadt, 2),
            round(new_commute_start, 2), round(new_commute_end, 2))


    
@app.callback(
    Output('county_selector_tab4', 'options'),
    Input('refresh_predictions_tab4', 'n_clicks')
)
def update_county_options(n_clicks):
    try:
        gpkg_file = './AI/Large_DataSet2.25_with_predictions.gpkg'
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



# ----------------------------------------------------------------
# 8.x. Callback to handle county‐selection, data‐editing, apply/reset predictions
@app.callback(
    Output('editable_gpkg_path', 'data'),
    Output('predictions_refresh', 'data'),
    [
        Input('county_selector_tab4', 'value'),
        Input('apply_updated_data',    'n_clicks'),
        Input('reset_predictions',     'n_clicks'),
        Input('model_selector_tab4',   'value'),
    ],
    [
        State('selected_census_tract',                'data'),
        State('editable_gpkg_path',                   'data'),
        State('input_DEMOGIDX_5',                     'value'),
        State('input_PEOPCOLORPCT',                   'value'),
        State('input_UNEMPPCT',                       'value'),
        State('input_pct_residential',                'value'),
        State('input_pct_industrial',                 'value'),
        State('input_pct_retail',                     'value'),
        State('input_pct_commercial',                 'value'),
        State('input_AADT',                           'value'),
        State('input_Commute_TripMiles_TripStart_avg','value'),
        State('input_Commute_TripMiles_TripEnd_avg',  'value'),
        State('predictions_refresh',                  'data')
    ]
)
def update_editable_gpkg_and_predictions(
    county_selector_value,
    apply_n_clicks,
    reset_n_clicks,
    model_file,
    selected_tract,
    gpkg_path,
    demogidx, peopcolorpct, unemppct,
    pct_residential, pct_industrial, pct_retail, pct_commercial,
    aadt, commute_start, commute_end,
    current_refresh
):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered = ctx.triggered[0]['prop_id'].split('.')[0]

    # 1) User just picked a county → create new editable GPKG
    if triggered == 'county_selector_tab4':
        if county_selector_value and len(county_selector_value) == 1:
            county = county_selector_value[0]
            new_file = copy_county_gpkg(county)
            return new_file, current_refresh
        else:
            raise PreventUpdate

    # 2) User clicked “Apply Updated Data”
    elif triggered == 'apply_updated_data':
        if apply_n_clicks and selected_tract and gpkg_path:
            # load, update the tract, save
            gdf = gpd.read_file(gpkg_path)
            idx = gdf[gdf['id'] == selected_tract].index
            if idx.empty:
                raise PreventUpdate
            # set each field
            gdf.loc[idx, 'DEMOGIDX_5']  = demogidx
            gdf.loc[idx, 'PEOPCOLORPCT'] = peopcolorpct
            gdf.loc[idx, 'UNEMPPCT']    = unemppct
            gdf.loc[idx, 'pct_residential'] = pct_residential
            gdf.loc[idx, 'pct_industrial']  = pct_industrial
            gdf.loc[idx, 'pct_retail']      = pct_retail
            gdf.loc[idx, 'pct_commercial']  = pct_commercial
            gdf.loc[idx, 'AADT']            = aadt
            gdf.loc[idx, 'Commute_TripMiles_TripStart_avg'] = commute_start
            gdf.loc[idx, 'Commute_TripMiles_TripEnd_avg']   = commute_end
            gdf.to_file(gpkg_path, driver="GPKG")
            import sys
            base, ext = os.path.splitext(gpkg_path)
            if model_file == 'AI2.py':
                suffix = '_with_gwr_predictions'
            else:
                suffix = '_with_predictions'
            output_file = f"{base}{suffix}{ext}"

            # build the command using the same Python interpreter
            cmd = [
                sys.executable,
                model_file,
                gpkg_path,
                output_file
            ]
            logger.debug(f"Running prediction command: {cmd}")

            try:
                proc = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.debug(f"{model_file} stdout:\n{proc.stdout}")
                logger.debug(f"{model_file} stderr:\n{proc.stderr}")
            except subprocess.CalledProcessError as e:
                # this will show you exactly what went wrong inside AI2.py
                logger.error(f"{model_file} failed with exit code {e.returncode}")
                logger.error(f"--- {model_file} stdout:\n{e.stdout}")
                logger.error(f"--- {model_file} stderr:\n{e.stderr}")
                # don’t let the app crash, but also don’t update the map
                raise PreventUpdate

            # if we get here, the script ran successfully
            new_refresh = (current_refresh or 0) + 1
            return gpkg_path, new_refresh

        else:
            raise PreventUpdate

    # 3) User clicked “Reset Predictions” → delete editable+pred files & remake
    elif triggered == 'reset_predictions':
        if reset_n_clicks:
            # remove the old editable gpkg
            if gpkg_path and os.path.exists(gpkg_path):
                os.remove(gpkg_path)
            # remove its predictions
            base, ext = os.path.splitext(gpkg_path or '')
            if model_file == 'AI2.py':
                suffix = '_with_gwr_predictions'
            else:
                suffix = '_with_predictions'
            pred_file = f"{base}{suffix}{ext}"
            if os.path.exists(pred_file):
                os.remove(pred_file)

            # re-copy the original for the currently selected county
            if county_selector_value and len(county_selector_value) == 1:
                new_file = copy_county_gpkg(county_selector_value[0])
            else:
                new_file = None

            return new_file, (current_refresh or 0) + 1
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
        suffix, col = "_with_gwr_predictions", "GWR_Prediction"
    else:
        suffix, col = "_with_predictions",     "Prediction"

    base, ext = os.path.splitext(gpkg_path)
    county_pred = f"{base}{suffix}{ext}"

    # load per‐county if it exists, otherwise global
    if os.path.exists(county_pred):
        gdf = gpd.read_file(county_pred)
    else:
        gdf = gpd.read_file(DEFAULT_PRED_FILES[model_file])

    if col not in gdf.columns:
        # nothing to store yet
        return None

    return gdf.loc[gdf['id'] == tract_id, col].iloc[0]

# ----------------------------------------------------------------
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
        suffix, col = "_with_gwr_predictions", "GWR_Prediction"
    else:
        suffix, col = "_with_predictions", "Prediction"

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
    census_data_file = os.path.join(data_folder, 'TractData.gpkg')  # GeoPackage file for Census Data

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

    app.run_server(debug=True)

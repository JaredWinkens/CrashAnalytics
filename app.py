import dash
from dash import dcc, html, Input, Output, State
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

# ----------------------------
# 1. Setup Logging
# ----------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------
# 2. Data Loading and Preprocessing
# ----------------------------

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
    For each feature, convert the geometry to geojson and add all attributes (except geometry)
    to the properties so that later you can use, for example, DEMOGIDX_5, PEOPCOLORPCT, etc.
    """
    census_polygons_by_county = {}
    try:
        gdf = gpd.read_file(file_path)
        logger.debug(f"Census GeoPackage loaded with {len(gdf)} records.")
        for idx, row in gdf.iterrows():
            county_name = row.get('CNTY_NAME')
            if pd.isna(county_name):
                logger.warning(f"Missing CNTY_NAME for row {idx}. Skipping.")
                continue
            county_name = county_name.strip().title()

            # Get the geometry from either 'geom' or 'geometry'
            geom_val = row.get('geom')
            if geom_val is None:
                geom_val = row.get('geometry')
            if geom_val is None:
                logger.error(f"Row {idx} does not have a geometry value. Skipping.")
                continue

            try:
                polygon_geojson = mapping(geom_val)
            except Exception as e:
                logger.error(f"Error converting geometry at row {idx}: {e}")
                continue

            # Include all attributes except the geometry columns.
            properties = row.to_dict()
            properties.pop('geom', None)
            properties.pop('geometry', None)
            polygon_geojson["properties"] = properties

            if county_name not in census_polygons_by_county:
                census_polygons_by_county[county_name] = []
            census_polygons_by_county[county_name].append(polygon_geojson)
        
        logger.debug(f"Total counties with polygons: {len(census_polygons_by_county)}.")
        return census_polygons_by_county

    except Exception as e:
        logger.error(f"Error loading Census data from .gpkg: {e}")
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
            style={'width': '100%'}
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
    
    return html.Div(controls, style={'width': '25%', 'float': 'left', 'padding': '20px', 'margin-top': '20px'})

def census_controls():
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
                    [{'label': county, 'value': county} for county in county_coordinates.keys()],
            value=['Albany'],
            multi=True,
            placeholder='Select one or more counties or All',
            style={'width': '100%'}
        ),
        html.Label("Select Census Attribute for Opacity:", style={'margin-top': '20px'}),
        dcc.Dropdown(
            id='census_attribute_selector',
            options=[
                {'label': 'Demographic Index', 'value': 'DEMOGIDX_5'},
                {'label': 'People of Color %', 'value': 'PEOPCOLORPCT'},
                {'label': 'Unemployment %', 'value': 'UNEMPPCT'},
                {'label': 'Residential %', 'value': 'pct_residential'},
                {'label': 'Industrial %', 'value': 'pct_industrial'},
                {'label': 'Retail %', 'value': 'pct_retail'},
                {'label': 'Commercial %', 'value': 'pct_commercial'},
                {'label': 'AADT Crash Rate', 'value': 'AADT Crash Rate'},
                {'label': 'VRU Crash Rate', 'value': 'VRU Crash Rate'},
                {'label': 'AADT', 'value': 'AADT'},
                {'label': 'Commute TripMiles Start Avg', 'value': 'Commute_TripMiles_TripStart_avg'},
                {'label': 'Commute TripMiles End Avg', 'value': 'Commute_TripMiles_TripEnd_avg'},
                {'label': 'Commute Biking and Walking Mile', 'value': 'Commute_BIKING_and_WALKING_Mile'},
                {'label': 'Commute Biking and Walking Mi 1', 'value': 'Commute_BIKING_and_WALKING_Mi_1'},
            ],
            value='DEMOGIDX_5'
        ),
        html.Div(id='census_color_legend', style={'margin-top': '20px'}),
        # New text box below the opacity bar:
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
    return html.Div(controls, style={'width': '25%', 'float': 'left', 'padding': '20px', 'margin-top': '20px'})

# ----------------------------
# 6. Define the Main Layout
# ----------------------------
app.layout = html.Div([
    dcc.Graph(id='scatter_map_tab3', style={'display': 'none'}),

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
            html.H1('Crash Data Analytics', style={
                'color': 'white', 'textAlign': 'center', 'lineHeight': '90px'
            }),
            # Right Image: NY.svg
            html.Img(src='/assets/NY.svg', style={
                'height': '128px', 'float': 'right', 'margin-right': '20px', 
                'margin-left': '40px', 'margin-top': '-150px'
            }),
        ], style={
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
    dcc.Download(id='download_data_tab3')   # For Census Data tab
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
        return html.Div([
            # ... existing code for tab-1 ...
            common_controls(
                'tab1',
                show_buttons=True,
                available_counties=available_counties,
                unique_weather=unique_weather,
                unique_light=unique_light,
                unique_road=unique_road
            ),
            html.Div(id='warning_message_tab1', style={'color': 'red', 'margin-left': '25%'}),
            dcc.Graph(
                id='scatter_map',
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
                style={'height': '80vh', 'width': '70%', 'position': 'fixed', 'top': '160px', 'right': '20px'},
                config={'modeBarButtonsToRemove': ['lasso2d'], 'displayModeBar': True, 'scrollZoom': True}
            ),
        ])

    elif tab == 'tab-2':
        return html.Div([
            # ... existing code for tab-2 ...
            common_controls(
                'tab2',
                show_buttons=True,
                available_counties=available_counties,
                unique_weather=unique_weather,
                unique_light=unique_light,
                unique_road=unique_road
            ),
            html.Div([
                html.Label('Adjust Heatmap Radius:', style={'font-weight': 'bold', 'margin-top': '20px'}),
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
            ], style={'width': '70%', 'margin': '20px auto 0 auto'}),
            dcc.Graph(
                id='heatmap_graph',
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
                style={'height': '80vh', 'width': '70%', 'position': 'fixed', 'top': '250px', 'right': '20px'},
                config={'modeBarButtonsToRemove': ['lasso2d'], 'displayModeBar': True, 'scrollZoom': True}
            ),
        ])

    elif tab == 'tab-3':
        return html.Div([
            census_controls(),
            html.Div(id='warning_message_tab3', style={'color': 'red', 'margin-left': '25%'}),
            dcc.Graph(
                id='scatter_map_tab3',
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
                style={'height': '80vh', 'width': '70%', 'position': 'fixed', 'top': '160px', 'right': '20px'},
                config={'modeBarButtonsToRemove': ['lasso2d'], 'displayModeBar': True, 'scrollZoom': True}
            ),
        ])

    elif tab == 'tab-4':  # NEW TAB: Predictions
        return html.Div([
            # Left-side controls (county selector and refresh button)
            html.Div([
                html.Label('Select County:'),  # New county selector
                dcc.Dropdown(
                    id='county_selector_tab4',
                    options=[],  # This will be updated via a callback below.
                    multi=True,
                    placeholder='Select county by CNTY_NAME'
                ),
                html.Br(),
                html.Label('Prediction Data Controls'),
                html.Button('Refresh Predictions', id='refresh_predictions_tab4', n_clicks=0)
            ], style={'width': '25%', 'float': 'left', 'padding': '20px', 'margin-top': '20px'}),
            html.Div(id='warning_message_tab4', style={'color': 'red', 'margin-left': '25%'}),
            dcc.Graph(
                id='predictions_map',
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
                style={'height': '80vh', 'width': '70%', 'position': 'fixed', 'top': '160px', 'right': '20px'},
                config={'modeBarButtonsToRemove': ['lasso2d'], 'displayModeBar': True, 'scrollZoom': True}
            )
        ])



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
def update_map_tab1(apply_n_clicks, clear_n_clicks, counties_selected, selected_data,
                    start_date, end_date, time_range, days_of_week,
                    gender, weather, light, road_surface, main_data_type, vru_data_type):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            triggered_input = 'initial_load'
        else:
            triggered_input = ctx.triggered[0]['prop_id'].split('.')[0]
        logger.debug(f"Triggered Input: {triggered_input}")

        df = get_county_data(counties_selected)
        if df.empty:
            if not counties_selected or 'All' in counties_selected:
                lat_center, lon_center = 40.7128, -74.0060
            else:
                lat_center = sum([county_coordinates[c]['lat'] for c in counties_selected]) / len(counties_selected)
                lon_center = sum([county_coordinates[c]['lon'] for c in counties_selected]) / len(counties_selected)
            fig = px.scatter_mapbox(
                pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
                lat='Latitude', lon='Longitude', zoom=10, mapbox_style="open-street-map"
            )
            fig.update_traces(marker=dict(opacity=0))
            return fig, None

        if triggered_input == 'apply_filter_tab1':
            effective_start_date = start_date
            effective_end_date = end_date
            logger.debug(f"Applying user-selected date range: {effective_start_date} to {effective_end_date}")

            # Apply the main filters.
            filtered_df = filter_data_tab1(
                df, effective_start_date, effective_end_date, time_range,
                days_of_week, gender, weather, light, road_surface, main_data_type, vru_data_type
            )

            # *** NEW: If a box has been drawn, filter to only those selected points ***
            if selected_data and 'points' in selected_data and selected_data['points']:
                selected_case_numbers = [point['customdata'][0] for point in selected_data['points']]
                filtered_df = filtered_df[filtered_df['Case_Number'].isin(selected_case_numbers)]
                logger.debug(f"Applied box selection: {len(filtered_df)} records remain after filtering by drawn box.")

            if filtered_df.empty:
                if not counties_selected or 'All' in counties_selected:
                    lat_center, lon_center = 40.7128, -74.0060
                else:
                    lat_center = sum([county_coordinates[c]['lat'] for c in counties_selected]) / len(counties_selected)
                    lon_center = sum([county_coordinates[c]['lon'] for c in counties_selected]) / len(counties_selected)
                fig = px.scatter_mapbox(
                    pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
                    lat='Latitude', lon='Longitude', zoom=10, mapbox_style="open-street-map"
                )
                fig.update_traces(marker=dict(opacity=0))
            else:
                if 'All' in counties_selected:
                    lat_center = df['Y_Coord'].mean()
                    lon_center = df['X_Coord'].mean()
                else:
                    lat_center = sum([county_coordinates[c]['lat'] for c in counties_selected]) / len(counties_selected)
                    lon_center = sum([county_coordinates[c]['lon'] for c in counties_selected]) / len(counties_selected)
                fig = px.scatter_mapbox(
                    filtered_df,
                    lat='Y_Coord', lon='X_Coord', zoom=10, mapbox_style="open-street-map",
                    hover_name='Case_Number',
                    hover_data={
                        'Crash_Date': True,
                        'Crash_Time': True,
                        'WeatherCon': True,
                        'LightCon': True,
                        'RoadSurfac': True
                    },
                    custom_data=['Case_Number']
                )
                fig.update_layout(mapbox_center={'lat': lat_center, 'lon': lon_center})

            logger.debug(f"Updated Scatter Map for counties {counties_selected} with {len(filtered_df)} records.")
            return fig, selected_data

        elif triggered_input == 'clear_drawing_tab1':
            # Clear the drawn box (selectedData) by reapplying filters without it.
            effective_start_date = '1900-01-01'
            effective_end_date = '1901-01-01'
            logger.debug(f"Clearing drawing and reapplying filters: {effective_start_date} to {effective_end_date}")
            filtered_df = filter_data_tab1(
                df, effective_start_date, effective_end_date, time_range,
                days_of_week, gender, weather, light, road_surface, main_data_type, vru_data_type
            )
            if filtered_df.empty:
                if not counties_selected or 'All' in counties_selected:
                    lat_center, lon_center = 40.7128, -74.0060
                else:
                    lat_center = sum([county_coordinates[c]['lat'] for c in counties_selected]) / len(counties_selected)
                    lon_center = sum([county_coordinates[c]['lon'] for c in counties_selected]) / len(counties_selected)
                fig = px.scatter_mapbox(
                    pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
                    lat='Latitude', lon='Longitude', zoom=10, mapbox_style="open-street-map"
                )
                fig.update_traces(marker=dict(opacity=0))
            else:
                if 'All' in counties_selected:
                    lat_center = df['Y_Coord'].mean()
                    lon_center = df['X_Coord'].mean()
                else:
                    lat_center = sum([county_coordinates[c]['lat'] for c in counties_selected]) / len(counties_selected)
                    lon_center = sum([county_coordinates[c]['lon'] for c in counties_selected]) / len(counties_selected)
                fig = px.scatter_mapbox(
                    filtered_df,
                    lat='Y_Coord', lon='X_Coord', zoom=10, mapbox_style="open-street-map",
                    hover_name='Case_Number',
                    hover_data={
                        'Crash_Date': True,
                        'Crash_Time': True,
                        'WeatherCon': True,
                        'LightCon': True,
                        'RoadSurfac': True
                    },
                    custom_data=['Case_Number']
                )
                fig.update_layout(mapbox_center={'lat': lat_center, 'lon': lon_center})
            logger.debug(f"Scatter Map after clearing drawing for counties {counties_selected} has {len(filtered_df)} records.")
            return fig, None

        else:
            # Initial load branch
            effective_start_date = '1900-01-01'
            effective_end_date = '1901-01-01'
            logger.debug(f"Initial load: Applying default date range {effective_start_date} to {effective_end_date}")
            filtered_df = filter_data_tab1(
                df, effective_start_date, effective_end_date, [0, 23],
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                'All', 'All', 'All', 'All', main_data_type, vru_data_type
            )
            if filtered_df.empty:
                if not counties_selected or 'All' in counties_selected:
                    lat_center, lon_center = 40.7128, -74.0060
                else:
                    lat_center = sum([county_coordinates[c]['lat'] for c in counties_selected]) / len(counties_selected)
                    lon_center = sum([county_coordinates[c]['lon'] for c in counties_selected]) / len(counties_selected)
                fig = px.scatter_mapbox(
                    pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
                    lat='Latitude', lon='Longitude', zoom=10, mapbox_style="open-street-map"
                )
                fig.update_traces(marker=dict(opacity=0))
            else:
                if 'All' in counties_selected:
                    lat_center = df['Y_Coord'].mean()
                    lon_center = df['X_Coord'].mean()
                else:
                    lat_center = sum([county_coordinates[c]['lat'] for c in counties_selected]) / len(counties_selected)
                    lon_center = sum([county_coordinates[c]['lon'] for c in counties_selected]) / len(counties_selected)
                fig = px.scatter_mapbox(
                    filtered_df,
                    lat='Y_Coord', lon='X_Coord', zoom=10, mapbox_style="open-street-map",
                    hover_name='Case_Number',
                    hover_data={
                        'Crash_Date': True,
                        'Crash_Time': True,
                        'WeatherCon': True,
                        'LightCon': True,
                        'RoadSurfac': True
                    },
                    custom_data=['Case_Number']
                )
                fig.update_layout(mapbox_center={'lat': lat_center, 'lon': lon_center})
            logger.debug(f"Initial Scatter Map for counties {counties_selected} has {len(filtered_df)} records.")
            return fig, None

    except Exception as e:
        logger.error(f"Error in update_map_tab1: {e}")
        if isinstance(counties_selected, list) and counties_selected and 'All' not in counties_selected:
            lat_center = sum([county_coordinates[c]['lat'] for c in counties_selected]) / len(counties_selected)
            lon_center = sum([county_coordinates[c]['lon'] for c in counties_selected]) / len(counties_selected)
        else:
            lat_center, lon_center = 40.7128, -74.0060
        fig = px.scatter_mapbox(
            pd.DataFrame({'Latitude': [lat_center], 'Longitude': [lon_center]}),
            lat='Latitude', lon='Longitude', zoom=10, mapbox_style="open-street-map"
        )
        fig.update_layout(
            annotations=[dict(
                text="An error occurred while updating the map.",
                showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=20)
            )]
        )
        fig.update_traces(marker=dict(opacity=0))
        return fig, None


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
        State('radius_slider_tab2', 'value'),  # Slider value now represents miles.
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
                        start_date, end_date, time_range, days_of_week, gender, weather, light, road_surface):
    try:
        zoom = 10  # Define a default zoom level.
        # Use a default center if needed.
        default_center = {'lat': 40.7128, 'lon': -74.0060}

        # If the button hasn't been pressed yet, show a default figure.
        if not apply_n_clicks:
            radius_pixels = convert_miles_to_pixels(radius_miles, zoom, default_center['lat'])
            fig = px.density_mapbox(
                pd.DataFrame({'Latitude': [default_center['lat']], 'Longitude': [default_center['lon']]}),
                lat='Latitude',
                lon='Longitude',
                radius=radius_pixels,
                center=default_center,
                zoom=zoom,
                mapbox_style="open-street-map",
                opacity=0.7
            )
            return fig

        # Get data for the selected counties.
        df = get_county_data(counties_selected)
        if df.empty:
            radius_pixels = convert_miles_to_pixels(radius_miles, zoom, default_center['lat'])
            fig = px.density_mapbox(
                pd.DataFrame({'Latitude': [default_center['lat']], 'Longitude': [default_center['lon']]}),
                lat='Latitude',
                lon='Longitude',
                radius=radius_pixels,
                center=default_center,
                zoom=zoom,
                mapbox_style="open-street-map",
                opacity=0.7
            )
            return fig

        # Apply filters to the data.
        filtered_df = filter_data_tab1(
            df, start_date, end_date, time_range,
            days_of_week, gender, weather, light, road_surface,
            main_data_type, vru_data_type
        )
        
        # Determine the center of the map.
        if 'All' in counties_selected:
            center_lat = df['Y_Coord'].mean()
            center_lon = df['X_Coord'].mean()
        else:
            center_lat = sum([county_coordinates[c]['lat'] for c in counties_selected]) / len(counties_selected)
            center_lon = sum([county_coordinates[c]['lon'] for c in counties_selected]) / len(counties_selected)
        
        # Convert the slider's miles to pixel radius.
        radius_pixels = convert_miles_to_pixels(radius_miles, zoom, center_lat)
        
        # Create the density mapbox heatmap.
        fig = px.density_mapbox(
            filtered_df,
            lat='Y_Coord',
            lon='X_Coord',
            radius=radius_pixels,
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
        return fig

    except Exception as e:
        logger.error(f"Error in update_heatmap_tab2: {e}")
        zoom = 10
        radius_pixels = convert_miles_to_pixels(radius_miles, zoom, default_center['lat'])
        fig = px.density_mapbox(
            pd.DataFrame({'Latitude': [default_center['lat']], 'Longitude': [default_center['lon']]}),
            lat='Latitude',
            lon='Longitude',
            radius=radius_pixels,
            center=default_center,
            zoom=zoom,
            mapbox_style="open-street-map",
            opacity=0.7
        )
        fig.update_layout(
            annotations=[{
                'text': "An error occurred while updating the heatmap.",
                'showarrow': False,
                'xref': "paper",
                'yref': "paper",
                'x': 0.5,
                'y': 0.5,
                'font': {'size': 20}
            }]
        )
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
    try:
        fig = go.Figure()

        # Determine selected counties.
        if isinstance(counties_selected, list):
            if 'All' in counties_selected:
                selected_counties = list(census_polygons_by_county.keys())
            else:
                selected_counties = counties_selected
        else:
            selected_counties = [counties_selected]

        # Gather all attribute values for normalization.
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
            min_val, max_val = 0, 1  # Fallback normalization

        poly_lats = []
        poly_lons = []

        # Add Census Polygon Traces with dynamic opacity.
        for county in selected_counties:
            polygons = census_polygons_by_county.get(county, [])
            if not polygons:
                alternate_key = county + " County"
                polygons = census_polygons_by_county.get(alternate_key, [])
            for poly in polygons:
                if poly.get('type') == 'Polygon':
                    coords = poly.get('coordinates', [])[0]
                    if coords[0] != coords[-1]:
                        coords = list(coords) + [coords[0]]
                    if len(coords) < 3:
                        continue
                    lons, lats = zip(*coords)
                    poly_lats.extend(lats)
                    poly_lons.extend(lons)
                    try:
                        value = float(poly["properties"].get(selected_attribute, 0))
                        norm = (value - min_val) / (max_val - min_val) if max_val > min_val else 1.0
                    except Exception:
                        norm = 0.5
                    opacity = 0.1 + 0.9 * norm
                    fig.add_trace(
                        go.Scattermapbox(
                            lat=list(lats),
                            lon=list(lons),
                            mode='lines',
                            fill='toself',
                            fillcolor=f'rgba(0,255,0,{opacity})',
                            line=dict(color='green', width=2),
                            showlegend=False,
                            hoverinfo='skip'
                        )
                    )
                elif poly.get('type') == 'MultiPolygon':
                    for polygon in poly.get('coordinates', []):
                        coords = polygon[0]
                        if coords[0] != coords[-1]:
                            coords = list(coords) + [coords[0]]
                        if len(coords) < 3:
                            continue
                        lons, lats = zip(*coords)
                        poly_lats.extend(lats)
                        poly_lons.extend(lons)
                        try:
                            value = float(poly["properties"].get(selected_attribute, 0))
                            norm = (value - min_val) / (max_val - min_val) if max_val > min_val else 1.0
                        except Exception:
                            norm = 0.5
                        opacity = 0.1 + 0.9 * norm
                        fig.add_trace(
                            go.Scattermapbox(
                                lat=list(lats),
                                lon=list(lons),
                                mode='lines',
                                fill='toself',
                                fillcolor=f'rgba(0,255,0,{opacity})',
                                line=dict(color='green', width=2),
                                showlegend=False,
                                hoverinfo='skip'
                            )
                        )

        # Determine map center from polygons.
        if poly_lats and poly_lons:
            center_lat = sum(poly_lats) / len(poly_lats)
            center_lon = sum(poly_lons) / len(poly_lons)
        else:
            center_lat, center_lon = county_coordinates.get('Albany', {'lat': 42.6526, 'lon': -73.7562}).values()

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center={'lat': center_lat, 'lon': center_lon},
                zoom=10
            ),
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
        )
        return fig

    except Exception as e:
        logger.error(f"Error in update_map_tab3: {e}")
        center_lat, center_lon = 40.7128, -74.0060
        fig = px.scatter_mapbox(
            pd.DataFrame({'Latitude': [center_lat], 'Longitude': [center_lon]}),
            lat='Latitude', lon='Longitude', zoom=10, mapbox_style="open-street-map"
        )
        fig.update_layout(
            annotations=[dict(
                text="An error occurred while updating the map.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                font=dict(size=20)
            )],
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
        )
        fig.update_traces(marker=dict(opacity=0))
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




@app.callback(
    Output('predictions_map', 'figure'),
    [Input('refresh_predictions_tab4', 'n_clicks'),
     Input('county_selector_tab4', 'value')]
)
def update_predictions_map(n_clicks, selected_counties):
    try:
        # Read the GeoPackage produced by your AI script
        gpkg_file = './AI/Large_DataSet2.25_with_predictions.gpkg'
        gdf = gpd.read_file(gpkg_file)
        if gdf.empty:
            fig = px.scatter_mapbox(
                pd.DataFrame({'Latitude': [40.7128], 'Longitude': [-74.0060]}),
                lat='Latitude', lon='Longitude', zoom=10,
                mapbox_style="open-street-map"
            )
            fig.update_traces(marker=dict(opacity=0))
            return fig

        # Filter by county selection if provided (using CNTY_NAME)
        if selected_counties:
            gdf = gdf[gdf['CNTY_NAME'].isin(selected_counties)]
        
        # Ensure each polygon has an ID
        gdf['id'] = gdf.index.astype(str)
        
        # Convert GeoDataFrame to GeoJSON dictionary
        geojson_str = gdf.to_json()
        geojson_dict = json.loads(geojson_str)
        for feature in geojson_dict["features"]:
            feature["properties"]["id"] = feature.get("id", None)
        
        # Calculate the center using the centroids of the polygons
        center_lat = gdf.geometry.centroid.y.mean()
        center_lon = gdf.geometry.centroid.x.mean()
        
        # Exclude the geometry column from hover data to avoid serialization issues
        hover_cols = [col for col in gdf.columns if col != "geometry"]
        
        # Create the choropleth map using the polygons
        fig = px.choropleth_mapbox(
            gdf,
            geojson=geojson_dict,
            locations='id',
            color='Prediction',  # Ensure this field exists in your GeoPackage
            featureidkey="properties.id",
            hover_data=hover_cols,
            center={'lat': center_lat, 'lon': center_lon},
            mapbox_style="open-street-map",
            zoom=10,
            opacity=0.5
        )
        return fig

    except Exception as e:
        logger.error(f"Error updating predictions map: {e}")
        fig = px.scatter_mapbox(
            pd.DataFrame({'Latitude': [40.7128], 'Longitude': [-74.0060]}),
            lat='Latitude', lon='Longitude', zoom=10,
            mapbox_style="open-street-map"
        )
        fig.update_traces(marker=dict(opacity=0))
        return fig


@app.callback(
    Output('county_selector_tab4', 'options'),
    Input('refresh_predictions_tab4', 'n_clicks')
)
def update_county_options(n_clicks):
    try:
        gpkg_file = './AI/Large_DataSet2.25_with_predictions.gpkg'
        gdf = gpd.read_file(gpkg_file)
        unique_counties = sorted(gdf['CNTY_NAME'].dropna().unique().tolist())
        options = [{'label': county, 'value': county} for county in unique_counties]
        return options
    except Exception as e:
        logger.error(f"Error updating county selector options: {e}")
        return []


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

    app.run_server(debug=True)

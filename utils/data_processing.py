import pandas as pd
import geopandas as gpd
from app_instance import logger
from dash.exceptions import PreventUpdate
import os
from shapely.geometry import mapping  

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
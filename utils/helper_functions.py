import math

def convert_miles_to_pixels(miles, zoom, center_latitude):
    """
    Convert a distance in miles to a pixel radius for a density map.

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

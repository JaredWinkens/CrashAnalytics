import math
import geopandas as gpd
import numpy as np
import pandas as pd
import fiona
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint
from shapely import points
from shapely.strtree import STRtree
import osmnx as ox
import uuid
import crash_rate_analysis.osm as osm

# Read the gdb file
gdb_path = "data/NYSDOT_GDB_AADT_2023/AADT_2023.gdb"
layers = fiona.listlayers(gdb_path)
print(f"Layers in {gdb_path}: {layers}")

# Convert to GeoDataFrame
layer_name = 'AADT_2023' # Replace with an actual layer name from your GDB
gdf = gpd.read_file(gdb_path, driver="OpenFileGDB", layer=layer_name)
gdf = gdf.to_crs(epsg=4326)
gdf = gdf.explode(index_parts=False)

gdf['unique_id'] = gdf.index #[uuid.uuid4() for _ in range(len(gdf))] # Generate UID for each row
gdf['center_lat'] = gdf.geometry.centroid.y
gdf['center_lon'] = gdf.geometry.centroid.x

# for i, col in enumerate(gdf.columns):
#     print(i,". ", col, gdf[col].iloc[0])

# for idx, row in gdf.iterrows():
#     print(idx,". ",row['unique_id'])
#     print("------------------------------")
#     print(row['geometry'])
#     if idx[0] >= 5:
#         break
# print(gdf.head())
# print(len(gdf))

def calculate_intersection_crash_rate(num_crashes_at_int: int, DEV: float, years: int):
    r_int = 0.0

    if DEV == 0.0 or years == 0.0:
        return r_int

    numerator = num_crashes_at_int * 1_000_000
    denominator = DEV * 365 * years

    r_int = numerator / denominator

    return r_int

def calculate_segment_crash_rate(num_crashes_on_seg: int, AADT: float, years: int, seg_length: float):
    r_seg = 0.0

    if AADT == 0.0 or years == 0.0 or seg_length == 0.0:
        return r_seg

    numerator = num_crashes_on_seg * 1_000_000
    denominator = AADT * 365 * years * seg_length

    r_seg = numerator / denominator

    return r_seg

def do_segment_analysis(counties: list, func_classes: list, study_period: int, filtered_crashes: pd.DataFrame):

    crash_geometry = points(filtered_crashes['X_Coord'], filtered_crashes['Y_Coord'])
    gdf_crashes = gpd.GeoDataFrame(filtered_crashes, geometry=crash_geometry, crs="EPSG:4326")

    gdf_lines = gdf
    if 'All' not in counties:
        gdf_lines = gdf_lines[gdf_lines['AADT_Stats_2023_Table_County'].isin(counties)]
    if 'All' not in func_classes:
        gdf_lines = gdf_lines[gdf_lines['AADT_Stats_2023_Table_Functional_Class'].isin(func_classes)]

    gdf_crashes_proj = gdf_crashes.to_crs(epsg=32618)
    gdf_lines_proj = gdf_lines.to_crs(epsg=32618)

    sjoin_result = gpd.sjoin_nearest(gdf_crashes_proj, gdf_lines_proj, how="left", max_distance=10, distance_col="distance_meters")

    crashes_on_multilinestrings_raw = sjoin_result[sjoin_result['distance_meters'].notna()]
    crash_counts_per_multilinestring = crashes_on_multilinestrings_raw.groupby('unique_id').size().reset_index(name='crash_count')

    gdf_lines = gdf_lines.merge(crash_counts_per_multilinestring, on='unique_id', how='left')
    gdf_lines['crash_count'] = gdf_lines['crash_count'].fillna(0).astype(int) # Fill NaN with 0 for lines with no crashes
    
    gdf_lines['crash_rate'] = 0.0
    gdf_lines['road_length_mi'] = 0.0
    segment_lats = []
    segment_lons = []
    segment_ids = []
    segment_crash_rates = []
    hover_text = []
    
    # Calculate crash rate for each segment
    for idx, row in gdf_lines.iterrows():
        x, y = row.geometry.xy
        
        seg_AADT = float(row['AADT_Stats_2023_Table_AADT']) # Annual Average Daily Traffic
        try: to_mile = float(row['AADT_Stats_2023_Table_To_Milepoint']) 
        except ValueError: to_mile = 0.0
        try: from_mile = float(row['AADT_Stats_2023_Table_From_Milepoint'])
        except ValueError: from_mile = 0.0
        seg_length_mi = to_mile - from_mile
        seg_crash_count = row['crash_count']
        crash_rate = calculate_segment_crash_rate(seg_crash_count, seg_AADT, study_period, seg_length_mi)
        gdf_lines.at[idx, 'crash_rate'] = crash_rate
        gdf_lines.at[idx, 'road_length_mi'] = seg_length_mi
        
        # Create custom hover text using other columns for this line
        hover_info = f"""
        ID: {row['unique_id']}<br>
        Description: {row['AADT_Stats_2023_Table_Description']}<br>
        Center: ({row['center_lat']:.4f}, {row['center_lon']:.4f})<br>
        Study Period (yrs): {study_period}<br>
        Length (mi): {seg_length_mi}<br>
        AADT: {seg_AADT}<br>
        Total Crashes: {seg_crash_count}<br>
        Estimated Crash Rate: {crash_rate}
        """
        
        segment_lons.extend(x)
        segment_lats.extend(y)
        segment_ids.extend([row['unique_id']] * len(x))
        segment_crash_rates.extend([row['crash_rate']] * len(x))
        hover_text.extend([hover_info] * len(x))

        # Add None to break the line between segments
        segment_lats.append(None)
        segment_lons.append(None)
        segment_crash_rates.append(None)
        segment_ids.append(None)
        hover_text.append(None)

    segment_trace = go.Scattermap(
        lat=segment_lats,
        lon=segment_lons,
        mode='lines',
        line=dict(
            width=4, # Adjust line thickness
            color='blue', # This is the key for coloring by crash rate
        ),
        text=hover_text, # Text to display on hover
        hoverinfo='text', # Show the custom text on hover
        name='Road Segments'
    )

    crashes_on_multilinestrings_to_plot = crashes_on_multilinestrings_raw.to_crs(epsg=4326)
    
    crash_trace = go.Scattermap(
        mode="markers",
        lon=crashes_on_multilinestrings_to_plot.geometry.x,
        lat=crashes_on_multilinestrings_to_plot.geometry.y,
        marker=dict(
            size=10,
            color='red', # Highlight crashes on lines in red
            opacity=0.9,
            symbol='circle'
        ),
        #hoverinfo="text",
        name="Crashes"
    )

    # Create the layout for the map
    layout = go.Layout(
        map=dict(
            style="open-street-map", # Or "carto-positron", "stamen-terrain", "stamen-watercolor", "stamen-toner"
            zoom=10,
            center=go.layout.map.Center(
                lat=gdf_lines.geometry.centroid.y.mean(),
                lon=gdf_lines.geometry.centroid.x.mean()
            )
        ),
        title=f'Segment Analysis of {", ".join(counties)}',
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    fig = go.Figure(data=[segment_trace, crash_trace], layout=layout)

    data = {
        'road_segments': gdf_lines,
        'crashes': crashes_on_multilinestrings_to_plot
    }

    return fig, data

def do_intersection_analysis(counties: list, county_coords: dict, func_classes: list, study_period: int, filtered_crashes: pd.DataFrame):

    radius_feet = 250
    radius_meters = radius_feet * 0.3048

    # Get filtered crash data
    crash_geometry = points(filtered_crashes['X_Coord'], filtered_crashes['Y_Coord'])
    gdf_crashes = gpd.GeoDataFrame(filtered_crashes, geometry=crash_geometry, crs="EPSG:4326")

    # Get filtered segment data
    gdf_lines = gdf
    if 'All' not in counties:
        gdf_lines = gdf_lines[gdf_lines['AADT_Stats_2023_Table_County'].isin(counties)]

    # Get filtered intersection data
    gdf_intersections_list = []
    for county in counties:
        area = f"{county} County, New York, USA" 
        gdf_intersections_list.append(osm.get_road_intersections(area))
    gdf_intersections = pd.concat(gdf_intersections_list, ignore_index=True)

    # filter intersections by class
    if 'All' not in func_classes:
        gdf_intersections = gdf_intersections[gdf_intersections['class'].isin(func_classes)]

    # Project data into a more percise EPSG format for analysis
    gdf_crashes_proj = gdf_crashes.to_crs(epsg=32618)
    gdf_lines_proj = gdf_lines.to_crs(epsg=32618)
    gdf_intersections_proj = gdf_intersections.to_crs(epsg=32618)

    # Get crashes within 250 feet of each intersection
    crashes_sjoin_result = gpd.sjoin_nearest(gdf_crashes_proj, gdf_intersections_proj, how="left", max_distance=75, distance_col="distance_meters")
    crashes_in_intersection_raw = crashes_sjoin_result[crashes_sjoin_result['distance_meters'].notna()]
    crash_counts_per_intersection = crashes_in_intersection_raw.groupby('node_id').size().reset_index(name='crash_count')
    gdf_intersections = gdf_intersections.merge(crash_counts_per_intersection, on='node_id', how='left')
    gdf_intersections['crash_count'] = gdf_intersections['crash_count'].fillna(0).astype(int) # Fill NaN with 0 for lines with no crashes
    crashes_in_intersections_to_plot = crashes_in_intersection_raw.to_crs(epsg=4326)

    # Get road segments inside each intersection
    intersections_buffered = gdf_intersections_proj.copy()
    intersections_buffered['geometry'] = intersections_buffered.geometry.buffer(radius_meters)
    joined_gdf = gpd.sjoin(gdf_lines_proj, intersections_buffered, how="inner", predicate="intersects")
    segment_ids_per_intersection = (joined_gdf.groupby(joined_gdf.index_right)['AADT_Stats_2023_Table_AADT'].apply(lambda x: list(x)).reset_index(name='nearby_segments_AADT'))
    intersections_gdf_with_segments = gdf_intersections.copy()
    intersections_gdf_with_segments['original_index'] = intersections_gdf_with_segments.index
    merged_gdf = pd.merge(intersections_gdf_with_segments,segment_ids_per_intersection,left_on='original_index',right_on='index_right',how='left')
    merged_gdf = merged_gdf.drop(columns=['original_index', 'index_right'])
    merged_gdf['nearby_segments_AADT'] = merged_gdf['nearby_segments_AADT'].apply(lambda x: x if isinstance(x, list) else [])

    merged_gdf['DEV'] = 0.0
    merged_gdf['crash_rate'] = 0.0

    hover_text = []
    # Get crash rate for every intersection
    for idx, row in merged_gdf.iterrows():
        int_crash_count = row['crash_count']
        segs_AADT = row['nearby_segments_AADT']
        if len(segs_AADT) == 0: int_DEV = 0.0
        else: int_DEV = sum(segs_AADT) / len(segs_AADT) # Daily Entering Volume (Average AADT of all segments inside intersection)
        merged_gdf.at[idx, 'DEV'] = int_DEV

        crash_rate = calculate_intersection_crash_rate(int_crash_count, int_DEV, study_period)
        merged_gdf.at[idx, 'crash_rate'] = crash_rate

        hover_info = f"""
        ID: {row['node_id']}<br>
        Coords: ({row['latitude']:.4f}, {row['longitude']:.4f})<br>
        Study Period (yrs): {study_period}<br>
        DEV: {int_DEV}<br>
        Total Crashes: {int_crash_count}<br>
        Estimated Crash Rate: {crash_rate}
        """

        hover_text.append(hover_info)

    all_lons = []
    all_lats = []
    hover_texts = [] # To store custom hover text

    for idx, row in gdf_lines.iterrows():
        # Get the MultiLineString geometry for the current row
        segment = row['geometry']

        # Create custom hover text using other columns for this line
        hover_info = f"""
        ID: {row['unique_id']}<br>
        Description: {row['AADT_Stats_2023_Table_Description']}<br>
        Center: ({row['center_lat']:.4f}, {row['center_lon']:.4f})<br>
        """
        x, y = segment.xy
        all_lons.extend(x)
        all_lats.extend(y)
        hover_texts.extend([hover_info] * len(x))
        all_lons.append(None)
        all_lats.append(None)
        hover_texts.append(None) # Keep hover_texts aligned

    segment_trace = go.Scattermap(
        mode="lines",
        lon=all_lons,
        lat=all_lats,
        hoverinfo="text",
        hovertext=hover_texts, # Use the custom hover text
        line=dict(width=2, color='green'), # Set a default color for all lines
        name="Road Segments"
    )

    intersection_trace = go.Scattermap(
        mode="markers",
        lon=gdf_intersections.geometry.x,
        lat=gdf_intersections.geometry.y,
        marker=dict(
            size=10,
            color='blue',
            opacity=0.9,
            symbol='circle',
        ),
        hoverinfo="text",
        hovertext=hover_text,
        name="Intersections"
    )

    crash_trace = go.Scattermap(
        mode="markers",
        lon=crashes_in_intersections_to_plot.geometry.x,
        lat=crashes_in_intersections_to_plot.geometry.y,
        marker=dict(
            size=10,
            color='red', # Highlight crashes on lines in red
            opacity=0.9,
            symbol='circle'
        ),
        #hoverinfo="text",
        name="Crashes in Intersection"
    )

    # Create the layout for the map
    layout = go.Layout(
        map=dict(
            style="open-street-map", # Or "carto-positron", "stamen-terrain", "stamen-watercolor", "stamen-toner"
            zoom=10,
            center=go.layout.map.Center(
                lat=gdf_lines.geometry.centroid.y.mean(),
                lon=gdf_lines.geometry.centroid.x.mean()
            )
        ),
        title=f'Intersection Analysis of {", ".join(counties)}',
        margin={"r":0,"t":40,"l":0,"b":0}
    )

    fig = go.Figure(data=[segment_trace, intersection_trace, crash_trace], layout=layout)
    
    data = {
        'road_segments': gdf_lines,
        'intersections': gdf_intersections,
        'crashes': crashes_in_intersections_to_plot,
        'merged': merged_gdf
    }

    return fig, data

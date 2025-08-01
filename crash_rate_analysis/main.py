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
import plotly.colors as pcolors
from datetime import datetime

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

    # Convert filtered crashes into GeoDataFrame
    crash_geometry = points(filtered_crashes['X_Coord'], filtered_crashes['Y_Coord'])
    gdf_crashes = gpd.GeoDataFrame(filtered_crashes, geometry=crash_geometry, crs="EPSG:4326")
    gdf_crashes = gdf_crashes.drop('Crash_Time', axis=1)
    gdf_crashes = gdf_crashes.drop('Crash_Date', axis=1)

    # Get filted line segements
    gdf_lines = gdf
    if 'All' not in counties:
        gdf_lines = gdf_lines[gdf_lines['AADT_Stats_2023_Table_County'].isin(counties)]
    if 'All' not in func_classes:
        gdf_lines = gdf_lines[gdf_lines['AADT_Stats_2023_Table_Functional_Class'].isin(func_classes)]

    # Project to a crs that uses meters
    gdf_crashes_proj = gdf_crashes.to_crs("EPSG:3857")
    gdf_lines_proj = gdf_lines.to_crs("EPSG:3857")

    sjoin_result = gpd.sjoin_nearest(gdf_crashes_proj, gdf_lines_proj, how="left", max_distance=10, distance_col="distance_meters")

    crashes_on_segments_raw = sjoin_result[sjoin_result['distance_meters'].notna()]
    crash_counts_per_segment = crashes_on_segments_raw.groupby('unique_id').size().reset_index(name='crash_count')

    gdf_lines = gdf_lines.merge(crash_counts_per_segment, on='unique_id', how='left')
    gdf_lines['crash_count'] = gdf_lines['crash_count'].fillna(0).astype(int) # Fill NaN with 0 for lines with no crashes
    
    gdf_lines['crash_rate'] = 0.0
    gdf_lines['road_length_mi'] = 0.0
    
    # Calculate crash rate for each segment
    for idx, row in gdf_lines.iterrows():
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

    num_categories = 10

    gdf_lines['crash_category_codes'] = pd.qcut(gdf_lines['crash_rate'], q=num_categories, duplicates='drop', labels=False)
    actual_categories_series = gdf_lines['crash_category_codes'].dropna().astype('category')
    actual_num_categories = len(actual_categories_series.cat.categories)
    gdf_lines['crash_category'] = pd.qcut(gdf_lines['crash_rate'], q=num_categories,
                                 labels=[f"Rate Bin {i+1}" for i in range(actual_num_categories)],
                                 duplicates='drop')
    
    Maps_colorscale = [
        [0.0, 'rgb(0, 153, 255)'],   # Bright Blue (very low traffic/crash)
        [0.2, 'rgb(0, 204, 0)'],     # Green (low traffic/crash)
        [0.5, 'rgb(255, 255, 0)'],   # Yellow (moderate traffic/crash)
        [0.75, 'rgb(255, 128, 0)'],  # Orange (high traffic/crash)
        [1.0, 'rgb(255, 0, 0)']      # Red (very high traffic/crash)
    ]
    colors_for_sampling = [item[1] for item in Maps_colorscale]
    colors = pcolors.sample_colorscale(colors_for_sampling, samplepoints=actual_num_categories)
    category_colors = {label: colors[i] for i, label in enumerate(gdf_lines['crash_category'].cat.categories)}

    category_data = {
        category: {'lats': [], 'lons': [], 'hover_texts': []}
        for category in gdf_lines['crash_category'].cat.categories
    }
    for idx, row in gdf_lines.iterrows():
        line_geometry = row['geometry']
        crash_category = row['crash_category']
        if pd.isna(crash_category):
            continue
        crash_rate = row['crash_rate']

        # Create custom hover text using other columns for this line
        hover_info = f"""
        ID: {row['unique_id']}<br>
        Description: {row['AADT_Stats_2023_Table_Description']}<br>
        Center: ({row['center_lat']:.4f}, {row['center_lon']:.4f})<br>
        Study Period (yrs): {study_period}<br>
        Length (mi): {row['road_length_mi']}<br>
        AADT: {row['AADT_Stats_2023_Table_AADT']}<br>
        Total Crashes: {row['crash_count']}<br>
        Estimated Crash Rate: {row['crash_rate']}
        """
        for coord in line_geometry.coords:
            category_data[crash_category]['lons'].append(coord[0])
            category_data[crash_category]['lats'].append(coord[1])
            category_data[crash_category]['hover_texts'].append(hover_info)

        # Add None to separate this line from the next within its category's lists
        category_data[crash_category]['lons'].append(None)
        category_data[crash_category]['lats'].append(None)
        category_data[crash_category]['hover_texts'].append(None)
 
    fig = go.Figure()

    for category_name in gdf_lines['crash_category'].cat.categories:
        data = category_data[category_name]
        if data['lons']: # Only add trace if there's data for the category
            fig.add_trace(go.Scattermap(
                mode="lines",
                lon=data['lons'],
                lat=data['lats'],
                line=dict(
                    width=3,
                    color=category_colors[category_name] # Single color for this trace
                ),
                hoverinfo="text",
                hovertext=data['hover_texts'],
                name=category_name, # Name for legend
                showlegend=False
            ))

    crashes_on_multilinestrings_to_plot = crashes_on_segments_raw.to_crs("EPSG:4326")
    
    # crash_trace = go.Scattermap(
    #     mode="markers",
    #     lon=crashes_on_multilinestrings_to_plot.geometry.x,
    #     lat=crashes_on_multilinestrings_to_plot.geometry.y,
    #     marker=dict(
    #         size=10,
    #         color='black',
    #         opacity=0.9,
    #         symbol='circle'
    #     ),
    #     #hoverinfo="text",
    #     name="Crashes"
    # )
    # fig.add_trace(crash_trace)

    valid_crash_rates = gdf_lines['crash_rate'].dropna()
    min_rate_overall = valid_crash_rates.min() if not valid_crash_rates.empty else 0.0
    max_rate_overall = valid_crash_rates.max() if not valid_crash_rates.empty else 1.0

    dummy_scatter_for_colorbar = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            size=0,
            color=gdf_lines['crash_rate'],
            colorscale=Maps_colorscale,
            cmin=min_rate_overall,
            cmax=max_rate_overall,
            showscale=True,
            colorbar=dict(
                title="Crash Rate",
                outlinecolor="black",
                outlinewidth=1,
                ticks="outside",
                ticklen=5,
                yanchor="top", y=1,
                xanchor="right", x=1,
                len=0.75,
                thickness=20
            )
        ),
        hoverinfo='none',
        showlegend=False
    )
    fig.add_trace(dummy_scatter_for_colorbar)
    
    fig.update_layout(
        map_style="open-street-map",
        map_zoom=10,
        map_center={"lat": gdf_lines.geometry.centroid.y.mean(), "lon": gdf_lines.geometry.centroid.x.mean()},
        margin={"r":0,"t":40,"l":0,"b":0},
        legend_title_text="Crash Rate Category (Bins)"
    )
    gdf_lines_json = gdf_lines.to_json()
    gdf_crashes_json = crashes_on_multilinestrings_to_plot.to_json()
    data = {
        'road_segments': gdf_lines_json,
        'crashes': gdf_crashes_json
    }

    return fig, data

def do_intersection_analysis(counties: list, func_classes: list, study_period: int, filtered_crashes: pd.DataFrame):

    radius_feet = 250
    radius_meters = radius_feet * 0.3048

    # Get filtered crash data
    crash_geometry = points(filtered_crashes['X_Coord'], filtered_crashes['Y_Coord'])
    gdf_crashes = gpd.GeoDataFrame(filtered_crashes, geometry=crash_geometry, crs="EPSG:4326")
    gdf_crashes = gdf_crashes.drop('Crash_Time', axis=1)
    gdf_crashes = gdf_crashes.drop('Crash_Date', axis=1)

    # Get filtered segment data
    gdf_lines = gdf
    if 'All' not in counties:
        gdf_lines = gdf_lines[gdf_lines['AADT_Stats_2023_Table_County'].isin(counties)]

    # Get filtered intersection data
    gdf_intersections_list: list[gpd.GeoDataFrame] = []
    for county in counties:
        area = f"{county} County, New York, USA" 
        gdf_intersections_list.append(osm.get_road_intersections(area))
    gdf_intersections: gpd.GeoDataFrame = pd.concat(gdf_intersections_list, ignore_index=True)

    # filter intersections by class
    if 'All' not in func_classes:
        gdf_intersections = gdf_intersections[gdf_intersections['class'].isin(func_classes)]

    # Project data into a more percise EPSG format for analysis
    gdf_crashes_proj = gdf_crashes.to_crs("EPSG:3857")
    gdf_lines_proj = gdf_lines.to_crs("EPSG:3857")
    gdf_intersections_proj = gdf_intersections.to_crs("EPSG:3857")

    # Create a buffer of 250 feet around each intersection
    intersection_buffers = gdf_intersections_proj.buffer(radius_meters)
    intersection_buffers_gdf = gpd.GeoDataFrame(gdf_intersections_proj, geometry=intersection_buffers, crs="EPSG:3857")
    
    # Get crashes within each intersection
    crashes_within_buffers = gpd.sjoin(gdf_crashes_proj, intersection_buffers_gdf, how='inner', predicate='intersects')
    #crash_counts = crashes_within_buffers.groupby('node_id').size().reset_index(name='crash_count')
    crash_analysis = crashes_within_buffers.groupby('node_id').agg(
        crash_count=('Case_Number', 'size'),
        crash_ids=('Case_Number', list)
    ).reset_index()
    
    segments_within_buffers = gpd.sjoin(gdf_lines_proj, intersection_buffers_gdf, how="inner", predicate="intersects")
    #segment_counts = segments_within_buffers.groupby('node_id').size().reset_index(name='segment_count')
    segment_analysis = segments_within_buffers.groupby('node_id').agg(
        segment_count=('unique_id', 'size'),
        segment_ids=('unique_id', list)
    ).reset_index()
    
    final_intersections_gdf = gdf_intersections_proj.merge(crash_analysis, on='node_id', how='left')
    final_intersections_gdf = final_intersections_gdf.merge(segment_analysis, on='node_id', how='left')
    
    final_intersections_gdf['crash_count'] = final_intersections_gdf['crash_count'].fillna(0).astype(int)
    final_intersections_gdf['segment_count'] = final_intersections_gdf['segment_count'].fillna(0).astype(int)

    # For the lists, we fill NaN with an empty list
    final_intersections_gdf['crash_ids'] = final_intersections_gdf['crash_ids'].apply(lambda x: x if isinstance(x, list) else [])
    final_intersections_gdf['segment_ids'] = final_intersections_gdf['segment_ids'].apply(lambda x: x if isinstance(x, list) else [])

    #crashes_in_intersections_to_plot = crashes_within_buffers.to_crs("EPSG:4326")

    final_intersections_gdf['DEV'] = 0.0
    final_intersections_gdf['crash_rate'] = 0.0

    hover_text_intersection = []
    # Get crash rate for every intersection
    for idx, row in final_intersections_gdf.iterrows():
        crash_count = row['crash_count']
        seg_count = row['segment_count']
        crash_ids = row['crash_ids']
        seg_ids = row['segment_ids']

        segs_df = gdf_lines[gdf_lines['unique_id'].isin(seg_ids)]
        segs_AADT = segs_df['AADT_Stats_2023_Table_AADT'].to_list()

        if len(segs_AADT) == 0: int_DEV = 0.0
        else: int_DEV = sum(segs_AADT) / len(segs_AADT) # Daily Entering Volume (Average AADT of all segments inside intersection)
        final_intersections_gdf.at[idx, 'DEV'] = int_DEV

        crash_rate = calculate_intersection_crash_rate(crash_count, int_DEV, study_period)
        final_intersections_gdf.at[idx, 'crash_rate'] = crash_rate

        hover_info = f"""
        ID: {row['node_id']}<br>
        Coords: ({row['latitude']:.4f}, {row['longitude']:.4f})<br>
        Study Period (yrs): {study_period}<br>
        DEV: {int_DEV}<br>
        Total Crashes: {crash_count}<br>
        Crash IDs: {crash_ids}<br>
        Total Segments: {seg_count}<br>
        Segment IDs: {seg_ids}<br>
        Estimated Crash Rate: {crash_rate}
        """

        hover_text_intersection.append(hover_info)

    all_lons = []
    all_lats = []
    hover_text_segment = [] # To store custom hover text

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
        hover_text_segment.extend([hover_info] * len(x))
        all_lons.append(None)
        all_lats.append(None)
        hover_text_segment.append(None) # Keep hover_texts aligned

    segment_trace = go.Scattermap(
        mode="lines",
        lon=all_lons,
        lat=all_lats,
        hoverinfo="text",
        hovertext=hover_text_segment, # Use the custom hover text
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
        hovertext=hover_text_intersection,
        name="Intersections"
    )

    crash_trace = go.Scattermap(
        mode="markers",
        lon=gdf_crashes.geometry.x,
        lat=gdf_crashes.geometry.y,
        marker=dict(
            size=10,
            color='red', # Highlight crashes on lines in red
            opacity=0.9,
            symbol='circle'
        ),
        hoverinfo="text",
        name="Crashes in Intersection"
    )

    intersection_buffers_gdf_wgs84 = intersection_buffers_gdf.to_crs("EPSG:4326")
    # Create the layout for the map
    layout = go.Layout(
        map=dict(
            style="open-street-map", # Or "carto-positron", "stamen-terrain", "stamen-watercolor", "stamen-toner"
            zoom=10,
            center=go.layout.map.Center(
                lat=gdf_lines.geometry.centroid.y.mean(),
                lon=gdf_lines.geometry.centroid.x.mean()
            ),
            # layers=[
            #     {
            #         "below": 'traces',
            #         "sourcetype": "geojson",
            #         "source": intersection_buffers_gdf_wgs84.__geo_interface__, # Convert GeoDataFrame to GeoJSON dict
            #         "type": "fill",
            #         "color": "rgba(0,100,0,0.3)" # Green with some transparency
            #     }
            # ]
        ),
        title=f'Intersection Analysis of {", ".join(counties)}',
        margin={"r":0,"t":40,"l":0,"b":0}
    )

    fig = go.Figure(data=[segment_trace, intersection_trace, crash_trace], layout=layout)
    
    gdf_lines_json = gdf_lines.to_json()
    gdf_crashes_json = gdf_crashes.to_json()
    final_intersections_gdf_json = final_intersections_gdf.to_json()

    data = {
        'road_segments': gdf_lines_json,
        'crashes': gdf_crashes_json,
        'intersections': final_intersections_gdf_json,
    }

    return fig, data

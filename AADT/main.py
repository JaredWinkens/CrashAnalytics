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
# Drop the temporary shapely point

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

def get_unique_col_values(column_name: str = 'AADT_Stats_2023_Table_Functional_Class') -> list:
    return gdf[column_name].dropna().unique().tolist()

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

def find_intersection_center_points(gdf: gpd.GeoDataFrame):
    
    segments: list[MultiLineString] = gdf['geometry']
    
    tree = STRtree(segments)

    intersection_centroids_data = []
    # intersection_centroids_data = {
    #     'pair_indices': [],
    #     'mls_a_wkt': [],
    #     'mls_b_wkt': [],
    #     'intersection_geometry_wkt': [],
    #     'centroid_wkt': []
    # }

    processed_pairs = set()

    for i, seg_a in enumerate(segments):
        possible_intersection_indices = tree.query(seg_a)
        for j in possible_intersection_indices:
            seg_b = segments[j]
            if i == j:
                continue
            if i < j:
                pair_indices = (i,j)
            else:
                pair_indices = (j,i)
            if pair_indices in processed_pairs:
                continue # Skip if already processed
            processed_pairs.add(pair_indices)
            if seg_a.intersects(seg_b):
                intersection_geometry = seg_a.intersection(seg_b)
                if not intersection_geometry.is_empty:
                    centroid_point = intersection_geometry.centroid
                    intersection_centroids_data.append({
                        'pair_indices': pair_indices,
                        'mls_a_wkt': seg_a.wkt,
                        'mls_b_wkt': seg_b.wkt,
                        'intersection_geometry_wkt': intersection_geometry.wkt,
                        'centroid_wkt': centroid_point.wkt,
                        'longitude': centroid_point.x,
                        'latitude': centroid_point.y
                    })
                    # intersection_centroids_data['pair_indices'].append(pair_indices)
                    # intersection_centroids_data['mls_a_wkt'].append(seg_a.wkt)
                    # intersection_centroids_data['mls_b_wkt'].append(seg_b.wkt)
                    # intersection_centroids_data['intersection_geometry_wkt'].append(intersection_geometry.wkt)
                    # intersection_centroids_data['centroid_wkt'].append(centroid_point)

    # if intersection_centroids_data:
    #     print("Intersection Centroids:")
    #     for data in intersection_centroids_data:
    #         print(f"  Intersection between MLS at index {data['pair_indices'][0]} and index {data['pair_indices'][1]}:")
    #         print(f"    Intersection Geometry: {data['intersection_geometry_wkt']}")
    #         print(f"    Center Point (Centroid): {data['centroid_wkt']}")
    #         print("-" * 30)
    # else:
    #     print("No intersections found between any of the MultiLineStrings.")
    return intersection_centroids_data

def get_intersections(gdf_lines: gpd.GeoDataFrame):
    roads_gdf = gdf_lines
    print(roads_gdf.head())
    #roads_gdf['geometry'] = roads_gdf['geometry'].buffer(0)

    intersections_sjoin = gpd.sjoin(roads_gdf, roads_gdf, how="inner", predicate="intersects", lsuffix="left", rsuffix="right")
    print(intersections_sjoin.head())
    intersections_sjoin = intersections_sjoin[intersections_sjoin['unique_id_left'] != intersections_sjoin['unique_id_right']]
    intersections_sjoin['pair_id'] = intersections_sjoin.apply(
        lambda row: tuple(sorted((row['unique_id_left'], row['unique_id_right']))), axis=1
    )
    intersections_sjoin = intersections_sjoin.drop_duplicates(subset=['pair_id'])
    intersection_geoms = []
    road_pairs = [] # To store the (id_left, id_right) pairs
    for idx, row in intersections_sjoin.iterrows():
        # Retrieve geometries directly using the unique IDs from the original DataFrame
        geom_left = roads_gdf.loc[roads_gdf['unique_id'] == row['unique_id_left'], 'geometry'].iloc[0]
        geom_right = roads_gdf.loc[roads_gdf['unique_id'] == row['unique_id_right'], 'geometry'].iloc[0]
        
        intersection_geoms.append(geom_left.intersection(geom_right))
        road_pairs.append(row['pair_id']) # Store the sorted ID pair
    intersections_gdf = gpd.GeoDataFrame(
        {'road_pair_ids': road_pairs}, # Store the unique ID pairs
        geometry=intersection_geoms,
        crs=roads_gdf.crs
    )
    point_intersections = intersections_gdf[
        intersections_gdf.geometry.apply(lambda geom: geom.geom_type in ['Point', 'MultiPoint'])
    ]
    exploded_points = point_intersections.explode(column='geometry')
    exploded_points = exploded_points[exploded_points.geom_type == 'Point']
    print("All intersecting geometries (can be Points, MultiPoints, or LineStrings):")
    print(intersections_gdf)
    print("\nOnly point intersections (exploded to individual points):")
    print(exploded_points)

    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lon=exploded_points.geometry.x,
        lat=exploded_points.geometry.y,
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=7, # Adjust as needed
        mapbox_center={"lat": gdf_lines.geometry.centroid.y.mean(), "lon": gdf_lines.geometry.centroid.x.mean()}, 
        margin={"r":0,"t":0,"l":0,"b":0},
        title="Intersections"
    )
    fig.show()

pbf_file_path = "data/new-york-latest.osm.pbf"
def get_intersections_osm(county: str):
    
    # counties_formatted = []
    # for county in counties:
    #     place_name = f"{county} County, New York, USA"
    #     counties_formatted.append(place_name)

    print(f"Downloading street network for {county} County, New York, USA")
    try:
        # 2. Download the street network graph for Albany
        # This will get all streets and their intersections (nodes)
        #north, south, east, west = 42.4072030, 42.8225420, -74.2646330, -73.6769420
        graph = ox.graph_from_place(f"{county} County, New York, USA", network_type="drive")

        print("Street network downloaded successfully.")

        # 3. Extract nodes (intersections) from the graph
        # Nodes in the graph represent intersections or dead ends
        nodes = graph.nodes(data=True)

        # Prepare a list to store intersection data
        intersections_data = {'node_id': [], 'latitude': [], 'longitude': []}

        print("Extracting latitude and longitude for each intersection...")

        # 4. Iterate through the nodes and get their latitude (y) and longitude (x)
        for node_id, data in nodes:
            lat = data['y']
            lon = data['x']
            intersections_data['node_id'].append(node_id)
            intersections_data['latitude'].append(lat)
            intersections_data['longitude'].append(lon)

        # 5. Store the data in a Pandas DataFrame for better organization
        geometry = points(intersections_data['longitude'], intersections_data['latitude'])
        intersections_gdf = gpd.GeoDataFrame(intersections_data, geometry=geometry, crs="EPSG:4326")

        print("\n--- Intersections Data ---")
        print(f"Found {len(intersections_gdf)} intersections in {county}.")
        print(intersections_gdf.head()) # Display the first few rows
        return intersections_gdf
        # Optional: Save to a CSV file
        # output_filename = "albany_intersections.csv"
        # df_intersections.to_csv(output_filename, index=False)
        # print(f"\nData saved to {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have an active internet connection and that 'osmnx' is installed.")
        print("You can install it using: pip install osmnx")
        return None

import matplotlib.cm as cm
import matplotlib.colors as colors

def get_color_map(lst: list[float], min, max):
    if min == max:
        # If all rates are the same, assign a neutral color (e.g., mid-range purple)
        return ["rgb(128,0,128)"] * len(lst)
    cmap = cm.get_cmap('coolwarm')
    norm = colors.Normalize(vmin=min, vmax=max)
    color_list = []
    for rate in lst:
        # Apply the normalizer to get a value between 0 and 1
        normalized_value = norm(rate)
        # Get the RGBA color from the colormap (values are 0-1 floats)
        rgba_color = cmap(normalized_value)

        # Convert RGBA (0-1) to RGB (0-255) integers
        r = int(rgba_color[0] * 255)
        g = int(rgba_color[1] * 255)
        b = int(rgba_color[2] * 255)

        # Format as an rgb string
        rgb_color_string = f"rgb({r},{g},{b})"
        color_list.append(rgb_color_string)

    return color_list

def do_segment_analysis(counties: list, func_classes: list, study_period: int, filtered_crashes: pd.DataFrame):

    crash_geometry = points(filtered_crashes['X_Coord'], filtered_crashes['Y_Coord'])
    gdf_crashes = gpd.GeoDataFrame(filtered_crashes, geometry=crash_geometry, crs="EPSG:4326")

    gdf_lines = gdf
    if 'All' not in counties:
        gdf_lines = gdf_lines[gdf_lines['AADT_Stats_2023_Table_County'].isin(counties)]
    if 'All' not in func_classes:
        gdf_lines = gdf_lines[gdf_lines['AADT_Stats_2023_Table_Functional_Class'].isin(func_classes)]

    print("Sample GeoDataFrame:")
    print(gdf_lines.head())
    print("\nGeoDataFrame CRS:", gdf_lines.crs)

    gdf_crashes_proj = gdf_crashes.to_crs(epsg=32618)
    gdf_lines_proj = gdf_lines.to_crs(epsg=32618)

    sjoin_result = gpd.sjoin_nearest(gdf_crashes_proj, gdf_lines_proj, how="left", max_distance=10, distance_col="distance_meters")

    crashes_on_multilinestrings_raw = sjoin_result[sjoin_result['distance_meters'].notna()]
    crash_counts_per_multilinestring = crashes_on_multilinestrings_raw.groupby('unique_id').size().reset_index(name='crash_count')
    print("\nCrash Counts per MultiLineString:")
    print(crash_counts_per_multilinestring)

    gdf_lines = gdf_lines.merge(crash_counts_per_multilinestring, on='unique_id', how='left')
    gdf_lines['crash_count'] = gdf_lines['crash_count'].fillna(0).astype(int) # Fill NaN with 0 for lines with no crashes

    print("\nGeoDataFrame with Crash Counts:")
    print(gdf_lines[['unique_id', 'crash_count']].head())
    
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

    return fig, gdf_lines

def get_ground_resolution(latitude, zoom_level):
    """
    Calculates the ground resolution (meters per pixel) at a given latitude and zoom level
    for the Web Mercator projection.
    """
    earth_radius_meters = 6378137
    tile_size_pixels = 256  # Standard for Mapbox/OpenStreetMap tiles

    # Convert latitude to radians
    lat_rad = math.radians(latitude)

    # Calculate circumference at the given latitude
    circumference_at_latitude = earth_radius_meters * 2 * math.pi * math.cos(lat_rad)

    # Calculate the number of pixels at this zoom level if the entire world was laid out
    # 2^zoom_level is the number of tiles along one dimension
    total_pixels_at_zoom = tile_size_pixels * (2 ** zoom_level)

    # Ground resolution in meters/pixel
    ground_resolution = circumference_at_latitude / total_pixels_at_zoom
    return ground_resolution

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
    if 'All' not in func_classes:
        gdf_lines = gdf_lines[gdf_lines['AADT_Stats_2023_Table_Functional_Class'].isin(func_classes)]

    # Get filtered intersection data
    gdf_intersections_list = []
    for county in counties: gdf_intersections_list.append(get_intersections_osm(county))
    gdf_intersections = pd.concat(gdf_intersections_list, ignore_index=True)

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
    
    return fig, merged_gdf

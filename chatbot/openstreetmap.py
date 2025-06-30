from google import genai
from google.genai import types
import requests
import json
import os
from pydantic import BaseModel
import urllib.parse
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union, linemerge, polygonize
import pandas as pd
import geopandas as gpd

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
GEN_MODEL = config['models']['2.0-flash']
API_KEY = config['general']['api_key']
gemini_client = genai.Client(api_key=API_KEY)
DEFAULT_BUFFER_DEG = 0.0001

class Response(BaseModel):
    location: str

def get_location_from_natural_lang(sentence):
    """
    Uses Gemini to extract a location from a natural language sentence.
    """
    prompt = f"""
    Extract the primary location from the following sentence and return it in JSON format with a 'location' key.
    If no specific location is found, return an empty string for the 'location' key.

    Sentence: '{sentence}'

    Example 1:
    Sentence: 'Find restaurants near Times Square.'
    Output: {{ "location": Times Square }}

    Example 2:
    Sentence: 'What's the weather like in London, UK?'
    Output: {{ "location": London, UK }}

    Example 3:
    Sentence: 'Tell me about historical events.'
    Output: {{ "location": "" }}
    """
    response = gemini_client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Response
        )
    )
    location: Response = response.parsed
    return location

def get_nominatim_data(location_query):
    """
    Queries Nominatim for geocoding data.
    """
    if not location_query:
        return None

    nominatim_url = f"https://nominatim.openstreetmap.org/search?q={location_query}&format=json&limit=1"
    headers = {
        'User-Agent': 'RoadSafetyChatbot/1.0 (winkenj@sunypoly.edu)' # Important: Provide a meaningful User-Agent
    }
    try:
        nominatim_response = requests.get(nominatim_url, headers=headers)
        nominatim_response.raise_for_status() # Raise an exception for HTTP errors
        data = nominatim_response.json()
        if data:
            return data[0] # Return the first (most relevant) result
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error querying Nominatim: {e}")
        return None

def get_overpass_geometry(osm_type, osm_id):
    """
    Queries Overpass API for the detailed geometry of a given OSM object (node, way, or relation).
    Constructs and returns a Shapely geometry object (Point, LineString, MultiLineString,
    Polygon, or MultiPolygon) based on the OSM type and retrieved data.
    Returns None if geometry cannot be retrieved or constructed.
    """
    if not osm_type or not osm_id:
        print("OSM type or ID is missing for Overpass query.")
        return None

    overpass_url = "https://overpass-api.de/api/interpreter"

    # Overpass QL query template:
    # For relations, we need to recurse down to get geometries of its members (ways/nodes).
    # For nodes/ways, we just get their own geometry.
    if osm_type == "relation":
        overpass_query = f"""
        [out:json][timeout:60];
        {osm_type}({osm_id});
        (._;>;);
        out geom;
        """
    else: # For node or way, just query the specific element
        overpass_query = f"""
        [out:json][timeout:60];
        {osm_type}({osm_id});
        out geom;
        """

    headers = {
        'User-Agent': 'CrashAnalyticsApp/1.0 (your.email@example.com)', # Use the same User-Agent as Nominatim
        'Content-Type': 'application/x-www-form-urlencoded' # Recommended for POST requests
    }

    payload = {'data': overpass_query}

    try:
        overpass_response = requests.post(overpass_url, data=payload, headers=headers)
        overpass_response.raise_for_status() # Raise an exception for HTTP errors
        overpass_data = overpass_response.json()
        
        elements = overpass_data.get('elements', [])

        if not elements:
            print(f"No elements found in Overpass response for {osm_type} ID {osm_id}.")
            return None

        # --- Handle Node geometry ---
        if osm_type == 'node':
            node_element = next((e for e in elements if e['type'] == 'node' and e['id'] == osm_id), None)
            if node_element and 'lat' in node_element and 'lon' in node_element:
                return Point(node_element['lon'], node_element['lat'])
            else:
                print(f"Node {osm_id} not found or missing geometry in Overpass response.")
                return None

        # --- Handle Way or Relation geometry (more complex processing for lines/polygons) ---
        elif osm_type == 'way' or osm_type == 'relation':
            all_line_strings = []
            
            # Iterate through ALL elements returned (especially crucial for relations,
            # which include their constituent ways).
            for element in elements:
                if element['type'] == 'way' and 'geometry' in element:
                    # Overpass geometry for ways is a list of lat/lon dicts.
                    # Convert to (lon, lat) tuples for Shapely.
                    coords = [(p['lon'], p['lat']) for p in element['geometry']]
                    if len(coords) >= 2: # A LineString needs at least 2 points
                        try:
                            line = LineString(coords)
                            if line.is_valid:
                                all_line_strings.append(line)
                            else:
                                # Attempt to fix invalid LineStrings (e.g., self-intersections, though rare for lines)
                                line_fixed = line.buffer(0)
                                if line_fixed.is_valid and not line_fixed.is_empty:
                                    all_line_strings.append(line_fixed)
                                else:
                                    print(f"Warning: Way {element['id']} formed an invalid or empty LineString after buffer(0). Skipping.")
                        except Exception as e:
                            print(f"Warning: Could not create LineString from way {element['id']}: {e}. Skipping.")

            if not all_line_strings:
                print(f"No valid LineStrings found for {osm_type} ID {osm_id} to form a boundary/line.")
                return None

            # Attempt to merge connected LineStrings into longer segments or multi-segments.
            merged_lines = linemerge(all_line_strings)

            # Try to polygonize first: if the merged lines form closed loops,
            # these represent areas (polygons). This is common for relations and closed ways.
            polygons = list(polygonize(merged_lines))

            if polygons:
                # If polygons were formed, combine them into a single (Multi)Polygon.
                # This handles cases like a city boundary made of multiple joined ways.
                unified_geometry = unary_union(polygons)
                if not unified_geometry.is_valid:
                    unified_geometry = unified_geometry.buffer(0) # Attempt to fix invalid polygons
                    if not unified_geometry.is_valid or unified_geometry.is_empty:
                        print("Warning: Unified polygon geometry is still invalid or empty after buffer(0). Returning None.")
                        return None
                return unified_geometry
            else:
                # If no polygons formed, it means the geometry is an open LineString (or MultiLineString).
                # This is typical for roads, rivers, etc., that are not closed areas.
                if not merged_lines.is_valid:
                    merged_lines = merged_lines.buffer(0) # Attempt to fix any invalidity in the merged lines
                    if not merged_lines.is_valid or merged_lines.is_empty:
                        print("Warning: Merged LineString geometry is still invalid or empty after buffer(0). Returning None.")
                        return None
                return merged_lines # Returns LineString or MultiLineString

        else:
            print(f"Unhandled OSM type: {osm_type} for ID {osm_id}. Returning None.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error querying Overpass API for {osm_type} ID {osm_id}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Overpass geometry processing for {osm_type} ID {osm_id}: {e}")
        return None

def create_crash_geodataframe(crash_reports):
    """
    Converts a list of crash report dictionaries into a GeoDataFrame.
    It assumes 'Lat' and 'Lon' keys are present for each crash report.
    The GeoDataFrame is set to EPSG:4326 (WGS84) CRS.
    """
    if not crash_reports:
        print("No crash reports provided to create GeoDataFrame.")
        return gpd.GeoDataFrame(columns=['CaseNumber', 'Lat', 'Lon', 'geometry'], crs="EPSG:4326")

    df = pd.DataFrame(crash_reports)
    # Create shapely Point objects from Lat/Lon columns.
    # IMPORTANT: Shapely Point constructor expects (longitude, latitude) or (x, y).
    try:
        geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    except KeyError as e:
        print(f"Error: Crash reports missing expected column for geometry creation: {e}. Ensure 'Lat' and 'Lon' keys are present.")
        return gpd.GeoDataFrame() # Return empty if essential keys are missing
    except Exception as e:
        print(f"An unexpected error occurred during geometry creation for crash reports: {e}")
        return gpd.GeoDataFrame()

    # Create GeoDataFrame and set its Coordinate Reference System (CRS) to WGS84.
    crashes_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return crashes_gdf

def check_crashes_in_boundary_geopandas(crashes_gdf, target_shapely_geometry, buffer_dist_deg=DEFAULT_BUFFER_DEG):
    """
    Checks which crash report coordinates in a GeoDataFrame fall within or on the given target geometry.
    For Point or LineString target geometries, a buffer is applied to convert them into polygons,
    allowing a consistent 'within' spatial join predicate.

    Args:
        crashes_gdf (geopandas.GeoDataFrame): GeoDataFrame containing crash reports (points).
        target_shapely_geometry (shapely.geometry.base.BaseGeometry): The target Shapely geometry
                                                                       (Point, LineString, MultiLineString,
                                                                        Polygon, or MultiPolygon).
        buffer_dist_deg (float): The buffer distance in degrees. Used for Point and LineString
                                 targets to define an "on or near" area.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing only the crash reports that
                                fall within/on the target geometry. Returns an empty GeoDataFrame
                                if the check cannot be performed.
    """
    if target_shapely_geometry is None:
        print("Error: No target geometry provided for checking.")
        return gpd.GeoDataFrame()

    # Determine the effective geometry for the spatial join.
    # If the target is a Point or LineString, we buffer it to create a polygon area.
    effective_geometry = target_shapely_geometry

    if isinstance(target_shapely_geometry, (Point, LineString, MultiLineString)):
        print(f"Buffering {target_shapely_geometry.geom_type} by {buffer_dist_deg} degrees for 'on or near' check.")
        try:
            effective_geometry = target_shapely_geometry.buffer(buffer_dist_deg)
            if not effective_geometry.is_valid:
                print(f"Warning: Buffered geometry is invalid for {target_shapely_geometry.geom_type}. Attempting to fix with buffer(0).")
                effective_geometry = effective_geometry.buffer(0)
                if not effective_geometry.is_valid or effective_geometry.is_empty:
                    print("Error: Buffered geometry remains invalid or empty. Cannot perform check. Returning empty GeoDataFrame.")
                    return gpd.GeoDataFrame()
        except Exception as e:
            print(f"Error buffering geometry {target_shapely_geometry.geom_type}: {e}. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame()
    
    elif isinstance(target_shapely_geometry, (Polygon, MultiPolygon)):
        # If the target is already a Polygon or MultiPolygon, use it directly.
        pass
    else:
        print(f"Unsupported geometry type for check: {target_shapely_geometry.geom_type}. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame()

    # Create a single-row GeoDataFrame for the effective target geometry.
    # Ensure it has the same CRS as the crash reports GeoDataFrame for correct spatial operations.
    try:
        boundary_gdf = gpd.GeoDataFrame(index=[0], geometry=[effective_geometry], crs="EPSG:4326")
    except Exception as e:
        print(f"Error creating boundary GeoDataFrame from effective geometry: {e}. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame()

    # Perform the spatial join.
    # 'how="inner"' means only keep rows from `crashes_gdf` that satisfy the predicate.
    # 'predicate="within"' checks if the point from `crashes_gdf` is within the polygon from `boundary_gdf`.
    try:
        crashes_in_boundary = gpd.sjoin(crashes_gdf, boundary_gdf, how="inner", predicate="within")

        # The sjoin adds columns from the right GeoDataFrame (boundary_gdf) and an 'index_right' column.
        # We drop these as they are typically not needed in the result.
        crashes_in_boundary = crashes_in_boundary.drop(columns=['index_right'], errors='ignore')
        
        return crashes_in_boundary
    except Exception as e:
        print(f"An error occurred during the spatial join: {e}. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame()
 
def get_filtered_data(crashes_gdf, extracted_location):
    # Convert crash reports to GeoDataFrame upfront
    #crashes_gdf = create_crash_geodataframe(sample_crash_reports_list)
    print("\nOriginal Crash GeoDataFrame (first 5 rows):")
    print(crashes_gdf.head())
    print(f"Total crashes in sample: {len(crashes_gdf)}\n")

    #extracted_location = get_location_from_natural_lang(user_query).location

    if extracted_location:
        nominatim_result = get_nominatim_data(extracted_location)
        if nominatim_result:
            target_osm_id = nominatim_result.get('osm_id')
            target_osm_type = nominatim_result.get('osm_type')
            print(f"Nominatim found '{nominatim_result.get('display_name')}' (ID: {target_osm_id}, Type: {target_osm_type})")

            location_shapely_geometry = get_overpass_geometry(target_osm_type, target_osm_id)
            if location_shapely_geometry:
                print(f"Retrieved Shapely geometry (Type: {location_shapely_geometry.geom_type}) for '{extracted_location}'.")
                crashes_inside_gdf = check_crashes_in_boundary_geopandas(crashes_gdf, location_shapely_geometry)
                print(f"\nCrashes found inside '{extracted_location}': {len(crashes_inside_gdf)}")

                if not crashes_inside_gdf.empty:
                    print(crashes_inside_gdf[['CaseNumber', 'Latitude', 'Longitude']])
                    return crashes_inside_gdf
                else:
                    print("No crashes found within this boundary.")
            else:
                print(f"Failed to get geometry from Overpass for '{extracted_location}'.")
        else:
            print(f"No Nominatim result for extracted location '{extracted_location}'.")
    else:
        print(f"No location extracted from sentence.")

# --- Main Execution Flow (Demonstration) ---

if __name__ == "__main__":
    # 1. Simulate crash reports data
    # (These coordinates are illustrative; some are near Utica, others far,
    # and some are close to a Genesee St road segment in Utica for testing.)
    sample_crash_reports_list = [
        {'CaseNumber': 101, 'Lat': 43.1000, 'Lon': -75.2500},  # Roughly inside Utica city boundary
        {'CaseNumber': 102, 'Lat': 43.0900, 'Lon': -75.2300},  # Roughly inside Utica city boundary
        {'CaseNumber': 103, 'Lat': 40.7128, 'Lon': -74.0060},  # New York City (definitely outside Utica)
        {'CaseNumber': 104, 'Lat': 43.1500, 'Lon': -75.5000},  # Possibly outside Utica, depends on exact boundary
        {'CaseNumber': 105, 'Lat': 43.1050, 'Lon': -75.2450},  # Another one roughly inside Utica
        {'CaseNumber': 106, 'Lat': 42.4250657, 'Lon': -78.1613997},  # Very close to Genesee St in Utica (for Way test)
        {'CaseNumber': 107, 'Lat': 42.4275410, 'Lon': -78.1621760}   # Also very close to Genesee St
    ]

    # Convert crash reports to GeoDataFrame upfront
    crashes_gdf = create_crash_geodataframe(sample_crash_reports_list)
    print("\nOriginal Crash GeoDataFrame (first 5 rows):")
    print(crashes_gdf.head())
    print(f"Total crashes in sample: {len(crashes_gdf)}\n")


    # --- Test Case 1: City Boundary (OSM Type: Relation) ---
    print("\n--- Test Case 1: City Boundary (e.g., Utica, NY - Relation) ---")
    user_sentence_city = "how many crashes happened in Utica, NY."
    extracted_location_city = get_location_from_natural_lang(user_sentence_city).location

    if extracted_location_city:
        nominatim_result_city = get_nominatim_data(extracted_location_city)
        if nominatim_result_city:
            target_osm_id_city = nominatim_result_city.get('osm_id')
            target_osm_type_city = nominatim_result_city.get('osm_type')
            print(f"Nominatim found '{nominatim_result_city.get('display_name')}' (ID: {target_osm_id_city}, Type: {target_osm_type_city})")

            location_shapely_geometry_city = get_overpass_geometry(target_osm_type_city, target_osm_id_city)
            if location_shapely_geometry_city:
                print(f"Retrieved Shapely geometry (Type: {location_shapely_geometry_city.geom_type}) for '{extracted_location_city}'.")
                crashes_inside_gdf_city = check_crashes_in_boundary_geopandas(crashes_gdf, location_shapely_geometry_city)
                print(f"\nCrashes found inside '{extracted_location_city}': {len(crashes_inside_gdf_city)}")
                if not crashes_inside_gdf_city.empty:
                    print(crashes_inside_gdf_city[['CaseNumber', 'Lat', 'Lon']])
                else:
                    print("No crashes found within this boundary.")
            else:
                print(f"Failed to get geometry from Overpass for '{extracted_location_city}'.")
        else:
            print(f"No Nominatim result for extracted location '{extracted_location_city}'.")
    else:
        print(f"No location extracted from sentence: '{user_sentence_city}'.")


    # --- Test Case 2: A Road (OSM Type: Way) ---
    print("\n--- Test Case 2: A Road (e.g., Seymour St, Houghton, NY - Way) ---")
    # Using Nominatim to find a specific road (Way)
    nominatim_result_road = get_nominatim_data("Seymour St, Houghton, NY")
    
    if nominatim_result_road and nominatim_result_road.get('osm_type') == 'way':
        target_osm_id_road = nominatim_result_road.get('osm_id')
        target_osm_type_road = nominatim_result_road.get('osm_type')
        extracted_location_road_name = nominatim_result_road.get('display_name')
        print(f"Nominatim found '{extracted_location_road_name}' (ID: {target_osm_id_road}, Type: {target_osm_type_road})")

        location_shapely_geometry_road = get_overpass_geometry(target_osm_type_road, target_osm_id_road)
        if location_shapely_geometry_road:
            print(f"Retrieved Shapely geometry (Type: {location_shapely_geometry_road.geom_type}) for '{extracted_location_road_name}'.")
            # For roads, use the default buffer or adjust as needed for "on or near"
            crashes_on_road_gdf = check_crashes_in_boundary_geopandas(crashes_gdf, location_shapely_geometry_road, buffer_dist_deg=DEFAULT_BUFFER_DEG)
            print(f"\nCrashes found on or near '{extracted_location_road_name}': {len(crashes_on_road_gdf)}")
            if not crashes_on_road_gdf.empty:
                print(crashes_on_road_gdf[['CaseNumber', 'Lat', 'Lon']])
            else:
                print("No crashes found on or near this road.")
        else:
            print(f"Failed to get geometry from Overpass for '{extracted_location_road_name}'.")
    else:
        print("Could not find a Way (road) for 'Genesee St, Utica, NY' or Nominatim result was unexpected (not a 'way').")


    # --- Test Case 3: A Specific Point/POI (OSM Type: Node) ---
    print("\n--- Test Case 3: A Specific Point/POI (e.g., Houghton Academy - Node) ---")
    # Using Nominatim to find a specific Point of Interest (Node)
    nominatim_result_node = get_nominatim_data("Houghton Academy")
    
    if nominatim_result_node and nominatim_result_node.get('osm_type') == 'node':
        target_osm_id_node = nominatim_result_node.get('osm_id')
        target_osm_type_node = nominatim_result_node.get('osm_type')
        extracted_location_node_name = nominatim_result_node.get('display_name')
        print(f"Nominatim found '{extracted_location_node_name}' (ID: {target_osm_id_node}, Type: {target_osm_type_node})")

        location_shapely_geometry_node = get_overpass_geometry(target_osm_type_node, target_osm_id_node)
        if location_shapely_geometry_node:
            print(f"Retrieved Shapely geometry (Type: {location_shapely_geometry_node.geom_type}) for '{extracted_location_node_name}'.")
            # For points, use a smaller buffer for "on or very near" (e.g., 0.00001 degrees is ~1 meter)
            crashes_near_node_gdf = check_crashes_in_boundary_geopandas(crashes_gdf, location_shapely_geometry_node, buffer_dist_deg=0.00001)
            print(f"\nCrashes found on or near '{extracted_location_node_name}': {len(crashes_near_node_gdf)}")
            if not crashes_near_node_gdf.empty:
                print(crashes_near_node_gdf[['CaseNumber', 'Lat', 'Lon']])
            else:
                print("No crashes found on or near this POI.")
        else:
            print(f"Failed to get geometry from Overpass for '{extracted_location_node_name}'.")
    else:
        print("Could not find a Node (POI) for 'Utica Union Station' or Nominatim result was unexpected (not a 'node').")
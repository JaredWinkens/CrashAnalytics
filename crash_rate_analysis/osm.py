import requests
import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from collections import defaultdict
import time


def get_osm_area_id(area_name):
    geolocator = Nominatim(user_agent="intersection-finder")
    location = geolocator.geocode(area_name, exactly_one=True)
    if not location:
        raise ValueError(f"Could not find area: {area_name}")
    
    osm_type = location.raw["osm_type"]
    osm_id = int(location.raw["osm_id"])

    if osm_type == "relation":
        area_id = 3600000000 + osm_id
    elif osm_type == "way":
        area_id = 2400000000 + osm_id
    elif osm_type == "node":
        area_id = 2000000000 + osm_id
    else:
        raise ValueError(f"Unsupported OSM type: {osm_type}")
    
    print(f"Retrieved OSM area ID: {area_id} for {area_name}")
    return area_id


def query_roads(area_id):
    overpass_url = "http://overpass-api.de/api/interpreter"
    headers = {
        'User-Agent': 'CrashAnalyticsApp/1.0 (your.email@example.com)', # Use the same User-Agent as Nominatim
        'Content-Type': 'application/x-www-form-urlencoded' # Recommended for POST requests
    }
    query = f"""
    [out:json][timeout:60];
    area({area_id})->.searchArea;
    (
      way["highway"](area.searchArea);
    );
    out body;
    >;
    out skel qt;
    """
    response = requests.post(overpass_url, data=query, headers=headers)
    response.raise_for_status()
    return response.json()


def extract_intersections(overpass_data):
    node_coords = {}
    node_usage = defaultdict(set)

    for element in overpass_data["elements"]:
        if element["type"] == "node":
            node_coords[element["id"]] = (element["lon"], element["lat"])

        elif element["type"] == "way" and "nodes" in element:
            for node_id in element["nodes"]:
                node_usage[node_id].add(element["id"])

    intersection_data = []
    id_counter = 0
    for node_id, ways in node_usage.items():
        if len(ways) >= 2:
            coord = node_coords.get(node_id)
            if coord:
                lon, lat = coord
                intersection_data.append({
                    "node_id": id_counter,
                    "class": f'{len(ways)}-way',
                    "latitude": lat,
                    "longitude": lon,
                    "geometry": Point(coord)
                })
                id_counter += 1
                
    gdf = gpd.GeoDataFrame(intersection_data, crs="EPSG:4326")
    return gdf


def get_road_intersections(area_name):
    area_id = get_osm_area_id(area_name)
    print("Querying roads from Overpass...")
    overpass_data = query_roads(area_id)
    print("Extracting and classifying intersections...")
    intersections_gdf = extract_intersections(overpass_data)
    return intersections_gdf


if __name__ == "__main__":
    area = "Albany County, New York, USA"  # Change this to your area of interest
    gdf = get_road_intersections("New York, USA")
    print(gdf['class'].dropna().unique().tolist())
    #gdf.to_file("intersections.geojson", driver="GeoJSON")

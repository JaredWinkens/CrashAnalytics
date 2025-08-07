from google import genai
from google.genai import types
from google.genai.errors import ServerError
import pandas as pd
from pydantic import BaseModel
import json
from shapely import points, Point
import geopandas as gpd
import requests
import base64
import cv2
import numpy as np

# --- Configuration ---
# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
API_KEY = config['general']['api_key'] 
GEN_MODEL = config['models']['2.5-flash']

client = genai.Client(api_key=API_KEY)
ANALYZER_ROLE = """
You are a traffic safety analyst. 
"""
class Insight(BaseModel):
    historical_trends: str
    environmental_influence: str

def format_response(response: Insight) -> str:
    return f"""
    **Historical Trends (AI-driven):** {response.historical_trends}
    
    **Environmental Influence (AI-driven):** {response.environmental_influence}
    """

def analyze_image_ai(image_bytes, image_metadata, crash_info, historical_data):
    try:
        response = client.models.generate_content(
            model=GEN_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=ANALYZER_ROLE,
                temperature=0.5,
                response_mime_type="application/json",
                response_schema=Insight,
            ),
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png'
                ),
                f"""
                A crash occured at the location in the attached 360 degree panoramic image.

                Note: The attached picture is not the actual picture of the crash scene.

                I am providing you with the following information to aid your analysis:
                - The image metadata:
                {image_metadata}
                - Details about the crash:
                {crash_info}
                - Details about crashes in the surrounding area (77 meter radius):
                {historical_data}
                
                Given all this information do the following:
                1. Explain any historical trends based on the crashes in the surrounding area (~100 tokens)
                2. Speculate on how the surrounding environment/infrustructure influenced the crash (~100 tokens)
                """
            ]
        )
        myinsights: Insight = response.parsed
        formatted_response = format_response(response=myinsights)
        return formatted_response
    except ServerError as e:
        raise e
    except Exception as e:
        raise e

def get_historical_crash_data(search_radius_meters: float, query_lat: float, query_lon: float, data: pd.DataFrame):
    
    geometry = points(data['X_Coord'], data['Y_Coord'])
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

    given_point_lat = query_lat
    given_point_lon = query_lon
    given_point = Point(given_point_lon, given_point_lat)
    given_point_gdf = gpd.GeoDataFrame(geometry=[given_point], crs="EPSG:4326")

    radius_meters = search_radius_meters

    gdf_projected = gdf.to_crs("EPSG:3857")
    given_point_projected = given_point_gdf.to_crs("EPSG:3857")

    buffer_polygon = given_point_projected.geometry.buffer(radius_meters).unary_union

    points_within_radius = gdf_projected[gdf_projected.intersects(buffer_polygon)]

    points_within_radius_original_crs = points_within_radius.to_crs("EPSG:4326")

    return points_within_radius_original_crs

def get_location_name(latitude, longitude):
    """
    Retrieves the location name (address) from latitude and longitude using Nominatim.

    Args:
        latitude: The latitude coordinate.
        longitude: The longitude coordinate.

    Returns:
        A string containing the location name (address) if found, or None if not found.
    """
    url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={latitude}&lon={longitude}&limit=1"
    headers = {
        'User-Agent': 'RoadSafetyChatbot/1.0 (winkenj@sunypoly.edu)' # Important: Provide a meaningful User-Agent
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        if 'display_name' in data:
            return data['display_name']
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return None

def get_street_view_image_metadata(latitude, longitude):
    """
    Fetches a single Google Street View image and returns its metadata.
    """
    metadata_url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={latitude},{longitude}&key={API_KEY}"
    try:
        response = requests.get(metadata_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    
def get_street_view_image_bytes(latitude, longitude, heading, size="640x480", fov=90, pitch=0):
    """
    Fetches a single Google Street View image and returns its bytes.
    """
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": f"{latitude},{longitude}",
        "heading": heading,
        "fov": fov,
        "pitch": pitch,
        "key": API_KEY
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors

        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
            return response.content
        else:
            print(f"Error: API did not return an image for heading {heading}. Content-Type: {content_type}. Response: {response.text[:200]}...") # Print first 200 chars
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Street View image for heading {heading}: {e}")
        return None

def stitch_street_view_images_from_lat_lon(latitude, longitude, headings, image_size, fov_value, output_format=".jpg"):
    """
    Downloads overlapping Street View images as bytes, stitches them,
    and returns the stitched panorama as bytes in the specified format.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        headings (list[int]): List of compass headings for images.
        image_size (str): Desired size of individual images (e.g., "640x480").
        fov_value (int): Field of view for individual images.
        output_format (str): Format for the output panoramic image (e.g., ".jpg", ".png").

    Returns:
        bytes: The stitched panoramic image in the specified format, or None if stitching fails.
    """
    raw_image_bytes_list = []
    print(f"Fetching {len(headings)} Street View images as bytes for {latitude},{longitude}...")
    for heading in headings:
        img_bytes = get_street_view_image_bytes(latitude, longitude, heading, size=image_size, fov=fov_value)
        if img_bytes:
            raw_image_bytes_list.append(img_bytes)
        else:
            print(f"Skipping heading {heading} due to download error.")

    if not raw_image_bytes_list:
        print("No images successfully downloaded for stitching.")
        return None

    # Decode bytes to OpenCV (NumPy array) format
    images_np = []
    for img_bytes in raw_image_bytes_list:
        # Convert bytes to a numpy array, then decode
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img_decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_decoded is not None:
            images_np.append(img_decoded)
        else:
            print("Warning: Could not decode an image. Skipping.")

    if len(images_np) < 2:
        print("Not enough valid images decoded for stitching.")
        return None

    print(f"Attempting to stitch {len(images_np)} images...")

    # Create a stitcher object
    if cv2.__version__.startswith('3.'):
        stitcher = cv2.createStitcher()
    else:
        stitcher = cv2.Stitcher_create()

    # Stitch the images
    status, panorama_np = stitcher.stitch(images_np)

    if status == cv2.Stitcher_OK:
        print("Stitching successful!")
        # Encode the stitched NumPy array back into bytes
        is_success, buffer = cv2.imencode(output_format, panorama_np)
        if is_success:
            image_bytes = bytes(buffer)
            return image_bytes
        else:
            print(f"Failed to encode stitched image to {output_format} bytes.")
            return None
    else:
        status_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images or better overlap for stitching.",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed. Check image overlap and quality.",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameters adjustment failed. Very little overlap or highly distorted images.",
        }
        error_message = status_messages.get(status, f"Stitching failed with unknown status code: {status}")
        print(f"Stitching failed: {error_message}")
        return None

def encode_image_to_base64_data_uri(image_bytes, format="jpeg"):
    """
    Encodes image bytes to a Base64 data URI for embedding in HTML.
    """
    base64_encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/{format};base64,{base64_encoded_image}"
import requests
import os
import concurrent.futures
from google import genai
from google.genai import types
from google.genai.errors import ServerError
import pandas as pd
from pydantic import BaseModel
import json

# --- Configuration ---
# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
API_KEY = config['general']['api_key'] 
GEN_MODEL = config['models']['2.5-flash']

client = genai.Client(api_key=API_KEY)
ANALYZER_ROLE = """
You are a traffic safety analyst. 

Keep your analysis brief (aim for ~150 tokens).

Prioritize clarity, brevity, and practical observations.
"""
class Insight(BaseModel):
    analysis: str

def format_response(response: Insight) -> str:
    return f"""
    **Analysis:** {response.analysis}
    """

def analyze_image_ai(image_bytes, crash_info):
    try:
        response = client.models.generate_content(
            model=GEN_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=ANALYZER_ROLE,
                temperature=0,
                response_mime_type="application/json",
                response_schema=Insight,
            ),
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png'
                ),
                f"""
                A crash occured at the location in the attached image.

                Here is some information about the crash: 
                {crash_info}
                
                Given the picture crash information see if you can draw any insights into the influence of the surrounding infrastructure/environment.
                """
            ]
        )
        print(response.text)
        myinsights: Insight = response.parsed
        formatted_response = format_response(response=myinsights)
        return formatted_response
    except ServerError as e:
        raise e
    except Exception as e:
        raise e

def get_street_view_image(latitude, longitude, image_size = "640x640", fov = 90, heading = 0, pitch = 0):
    location = f"{latitude},{longitude}"
    # --- Construct the API URL ---
    BASE_URL = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": image_size,
        "location": location,
        "fov": fov,
        "heading": heading,
        "pitch": pitch,
        "key": API_KEY
    }

    # --- Make the API request ---
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Street View image: {e}")
        if response.status_code == 400:
            print("Check your API key, parameters, or location. A 400 error often means a bad request.")
        elif response.status_code == 403:
            print("API Key might be invalid or not enabled for the Street View Static API, or billing is not enabled.")
        print(f"Response content: {response.text}") # Print response content for debugging
    
    return response

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
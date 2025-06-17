import concurrent.futures
from google import genai
from google.genai import types
from google.genai.errors import ServerError
import pandas as pd
from pydantic import BaseModel
import json

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
API_KEY = config['general']['api_key'] 
GEN_MODEL = config['models']['2.0-flash']

client = genai.Client(api_key=API_KEY)


ANALYZER_ROLE = """
You are a traffic safety analyst. 

Keep your analysis brief (aim for ~300 tokens).

Prioritize clarity, brevity, and practical observations.
"""

def call_with_timeout(func, timeout, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Function '{func.__name__}' timed out after {timeout} seconds.")
            return f"Function '{func.__name__}' timed out after {timeout} seconds."
        except Exception as e:
            print(f"Function '{func.__name__}' raised an exception: {e}")
            return f"Function '{func.__name__}' raised an exception: {e}"
        
def get_insights(image_bytes) -> str:

    response = call_with_timeout(generate_response, 60, image_bytes)

    return response

class Insight(BaseModel):
    key_hotspots: str
    observed_patterns: str
    inferred_causes: str

def format_response(response: Insight) -> str:
    return f"""
    - **Key Hotspots:** {response.key_hotspots}\n\n
    - **Observed Patterns:** {response.observed_patterns}\n\n
    - **Inferred Causes:** {response.inferred_causes}\n\n
    """

def generate_response(image) -> str:
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
                    data=image,
                    mime_type='image/png'
                ),
                "Analyze the provided crash density heatmap."
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

def main():
    pass

if __name__ == "__main__":
    main()
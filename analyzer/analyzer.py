from google import genai
from google.genai import types
from google.genai.errors import ServerError
from analyzer.config import *

client = genai.Client(api_key="AIzaSyBQ2Ca6HSly3DdXo4e35Nd1PjoroVSyFzs")

analyzer_role = """
You are a data analysis assistant. Your task is to analyze traffic crash records and provide clear, actionable insights based on the input dataset. 
The dataset contains rows of crash events with features such as time, location, weather, lighting, and severity.

Format your output using Markdown with headers, bullet points, and emphasis for readability.

Avoid repeating the raw data. Focus on trends and insights based on aggregation or correlation.
"""

def get_insights(data) -> str:
    result = ""

    try:
        response = client.models.generate_content(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=analyzer_role,
                temperature=0,
            ),
            contents=[str(data)]
        )
        result = response.text
    except ServerError as e:
        result = str(e)
    except Exception as e:
        result = str(e)

    return result

def main():
    pass

if __name__ == "__main__":
    main()
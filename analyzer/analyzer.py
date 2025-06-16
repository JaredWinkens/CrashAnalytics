import concurrent.futures
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from analyzer.analyzer_config import *
import pandas as pd

client = genai.Client(api_key=API_KEY)

import pandas as pd

def generate_crash_data_summary(df: pd.DataFrame) -> dict:
    summary = {}

    # Basic shape and missing values
    summary["row_count"] = len(df)
    summary["column_count"] = len(df.columns)
    summary["missing_values"] = df.isnull().sum().to_dict()

    # Temporal analysis
    df['Crash_Date'] = pd.to_datetime(df['Crash_Date'], errors='coerce')
    summary["crashes_by_year"] = df['Crash_Date'].dt.year.value_counts().sort_index().to_dict()
    summary["crashes_by_month"] = df['Crash_Date'].dt.month.value_counts().sort_index().to_dict()
    summary["crashes_by_dayofweek"] = df['Crash_Date'].dt.day_name().value_counts().to_dict()

    # Crash time distribution
    summary["crashes_by_hour"] = pd.to_numeric(df["Crash_Time"], errors="coerce").dropna().astype(int).value_counts().sort_index().to_dict()

    # Severity breakdown
    summary["severity_distribution"] = df["SeverityCategory"].value_counts().to_dict()

    # Crash type breakdown
    summary["crash_type_distribution"] = df["Crash_Type"].value_counts().to_dict()

    # Environmental conditions
    summary["weather_conditions"] = df["WeatherCon"].value_counts().to_dict()
    summary["light_conditions"] = df["LightCon"].value_counts().to_dict()
    summary["road_surface_conditions"] = df["RoadSurfac"].value_counts().to_dict()

    # Demographic info
    #summary["sex_distribution"] = df["Sex"].value_counts().to_dict()

    # Geospatial spread
    summary["x_coord_range"] = [df["X_Coord"].min(), df["X_Coord"].max()]
    summary["y_coord_range"] = [df["Y_Coord"].min(), df["Y_Coord"].max()]
    summary["crash_density_centroid"] = {
        "x": df["X_Coord"].mean(),
        "y": df["Y_Coord"].mean()
    }

    return summary

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
        
def get_insights(filters: dict, filtered_data: pd.DataFrame, original_data: pd.DataFrame) -> str:

    original_data_summary = generate_crash_data_summary(original_data)
    filtered_data_summary = generate_crash_data_summary(filtered_data)

    response = call_with_timeout(generate_response, 60, filters, filtered_data_summary, original_data_summary)

    return response

def generate_response(filters: dict, filtered_data_summary: dict, original_data_summary: dict) -> str:
    try:
        response = client.models.generate_content(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=ANALYZER_ROLE,
                temperature=0,
            ),
            contents=[
                "Filtered Dataset Summary: ", str(filtered_data_summary),
                "Full Dataset Summary: ", str(original_data_summary),
                "Filters Used: ", str(filters), 
            ]
        )
        return response.text
    except ServerError as e:
        raise e
    except Exception as e:
        raise e

def main():
    pass

if __name__ == "__main__":
    main()
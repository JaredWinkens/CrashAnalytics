from google import genai
from google.genai import types
from google.genai.errors import ServerError
from analyzer.config import *
import pandas as pd

client = genai.Client(api_key="AIzaSyBQ2Ca6HSly3DdXo4e35Nd1PjoroVSyFzs")

analyzer_role = """
You are a traffic crash data analyst. Your task is to provide clear, in-depth, and meaningful insights about a **filtered dataset** of traffic crash data. You will be provided with:

- A **statistical summary** of the filtered dataset  
- A **summary of the original, unfiltered dataset**  
- A description of the **filters applied by the user**  

Your primary goal is to **analyze and interpret the filtered dataset**, using the unfiltered data and user filters only as **contextual support**. Do **not** focus on directly comparing or contrasting the filtered and unfiltered datasets.

Instead, use the unfiltered dataset to better understand what makes the filtered data noteworthy or unusual, and to provide meaningful baselines or expectations where appropriate. Pay particular attention to how the **filters may have shaped the data**. 
Speculate thoughtfully on how the selected filters could be influencing the observed patterns, and how different or additional filters might lead to different insights.

Always center the analysis on the **filtered dataset itself**. Provide actionable observations, trends, correlations, or anomalies that emerge from the filtered data. Use statistical reasoning and domain-relevant logic where applicable.

Output should be clear, concise, and structured for decision-making and further investigation.
"""

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
    summary["sex_distribution"] = df["Sex"].value_counts().to_dict()

    # Geospatial spread
    summary["x_coord_range"] = [df["X_Coord"].min(), df["X_Coord"].max()]
    summary["y_coord_range"] = [df["Y_Coord"].min(), df["Y_Coord"].max()]
    summary["crash_density_centroid"] = {
        "x": df["X_Coord"].mean(),
        "y": df["Y_Coord"].mean()
    }

    return summary


def get_insights(filters: dict, filtered_data: pd.DataFrame, original_data: pd.DataFrame) -> str:

    original_data_summary = generate_crash_data_summary(original_data)
    filtered_data_summary = generate_crash_data_summary(filtered_data)

    result = ""
    try:
        response = client.models.generate_content(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=analyzer_role,
                temperature=0,
            ),
            contents=[
                "Filtered Dataset Summary: ", str(filtered_data_summary),
                "Full Dataset Summary: ", str(original_data_summary),
                "User Filters: ", str(filters),  
            ]
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
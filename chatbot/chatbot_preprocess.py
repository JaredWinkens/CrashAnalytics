import pandas as pd
import sqlite3
import json
import chromadb
from chromadb.utils import embedding_functions
from tqdm.auto import tqdm
from google import genai
from google.genai import types

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
API_KEY = config['general']['api_key']
CSV_FILE = config['paths']['csv_file']
CHROMADB_PATH = config['paths']['chromadb_path']
CHROMA_COLLECTION = config['general']['chroma_collection']
DB_FILE = config['paths']['db_file']
DB_TABLE_NAME = config['general']['db_table_name'] 
EMBEDDING_MODEL = config['models']['004'] 

relevant_old_cols = [
    # Identifiers & Time
    'CaseNumber', 'CaseYear', 'CrashDate', 'CrashTimeF',
    # Location & Environment
    'CNTY_NAME', 'CityTownNa', 'OnStreet', 'ClosestCro',
    'LightCondi', 'WeatherCon', 'RoadwayCha', 'RoadSurfac',
    'POSTED_SPE', 'TrafficCon', 'TrafficWay', 'Intersecti',
    # Crash Details & Severity
    'CrashType', 'CollisionT', 'CrashSever', 'MaxInjuryS',
    'NumberOfFa', 'NumberOfIn', 'NumberOfSe', 'NumberOfOt',
    'NumberOfVe', 'Commercial', 'isLargeTru',
    # Parties Involved & Contributing Factors
    'VehicleTyp', 'VehiclyBod', 'VehicleBod', 
    'PreCrashAc', 'PreCrash_1', 'ApparentFa',
    'PersonType', 'PersonInju', 'DriverAgeV',
    # Administrative
    'ReportingA'
]

relevant_new_col = [
    # Identifiers & Time
    'crash_case_number', 'crash_case_year', 'crash_date', 'crash_time_formatted',
    # Location & Environment
    'county_name', 'city_town_name', 'on_street_name', 'closest_cross_street',
    'light_condition', 'weather_condition', 'roadway_character', 'road_surface_condition',
    'posted_speed_limit', 'traffic_control_device', 'trafficway_description', 'intersection_related',
    # Crash Details & Severity
    'crash_type_description', 'collision_type', 'crash_severity', 'maximum_injury_severity',
    'number_of_fatalities', 'number_of_injuries', 'number_of_severe_injuries', 'number_of_other_injuries',
    'number_of_vehicles_involved', 'commercial_vehicle_involved', 'is_large_truck_involved',
    # Parties Involved & Contributing Factors
    'vehicle_type', 'vehicle_body_type', 'vehicle_body_type_detailed',
    'pre_crash_action', 'pre_crash_action_detailed', 'apparent_contributing_factor',
    'person_type', 'person_injury_severity', 'driver_age_vehicle_1',
    # Administrative
    'reporting_agency'
]

def load_data(csv_file: str, sample_size: int = 1500):

    # Read CSV file into dataframe
    df = pd.read_csv(csv_file, header=0, parse_dates=['CrashDate'], usecols=relevant_old_cols)

    # Give rows more descriptive names
    col_rename = {}
    for i in range(len(relevant_old_cols)): 
        col_rename[relevant_old_cols[i]] = relevant_new_col[i]
    print(col_rename)
    df = df.rename(columns=col_rename)

    # Select n random rows
    df_sampled = df.sample(n=sample_size, random_state=42)
    print(df_sampled)

    return df_sampled

def populate_sqldb(df: pd.DataFrame) -> None:
    conn = sqlite3.connect(DB_FILE)
    df.to_sql(DB_TABLE_NAME, conn, if_exists='replace', index=False)
    conn.close()
    print("Database Created Successfully")

def get_embedding(text: str):
    client = genai.Client(api_key=API_KEY)
    tokens = text.split()
    try:
        # Gemini's embedding model has a context limit per call.
        # Ensure 'text' is not excessively long.
        if not text or len(tokens) > 8000: # Example limit, adjust based on model. max is 8192 tokens for 001
            print(f"Warning: Text too long or empty for embedding. Truncating or skipping: {text[:100]}...")
            text = str(tokens[:8000]) # Truncate if too long
            if not text: return None
        response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                task_type="retrieval_document",
                )
            )
        return response.embeddings[0].values
    except Exception as e:
        print(f"Error generating embedding for text: '{text[:50]}...' - {e}")
        return None

def create_narrative(row: pd.Series, available_columns: list) -> str:
    narrative_parts = []

    # Helper to safely get and format a value
    def get_val(col, default="unknown", formatter=str):
        if col in available_columns and pd.notna(row.get(col)):
            try:
                return formatter(row.get(col))
            except (ValueError, TypeError):
                return default # Return default if formatting fails
        return default
    
    # 1. Core Crash Information (Date, Time, Location)
    date_str = get_val('crash_date', "unknown date", lambda x: pd.to_datetime(x).strftime('%Y-%m-%d'))
    time_str = get_val('crash_time_formatted', "unknown time")
    city_str = get_val('city_town_name', "an unknown city/town")
    county_str = get_val('county_name', "an unknown county")
    on_street_str = get_val('on_street_name', "an unnamed street")
    cross_street_str = get_val('closest_cross_street', None)

    intro_sentence = f"On {date_str} at {time_str}, a crash occurred in {city_str}, {county_str}, on {on_street_str}"
    if cross_street_str:
        intro_sentence += f" near {cross_street_str}."
    else:
        intro_sentence += "."
    narrative_parts.append(intro_sentence)

    # 2. Environmental & Roadway Conditions
    env_parts = []
    if get_val('light_condition', None) != None:
        env_parts.append(f"light conditions were {get_val('light_condition').lower()}")
    if get_val('weather_condition', None) != None:
        env_parts.append(f"weather was {get_val('weather_condition').lower()}")
    if get_val('road_surface_condition', None) != None:
        env_parts.append(f"road surface was {get_val('road_surface_condition').lower()}")
    if env_parts:
        narrative_parts.append(f"Environmental factors: {', '.join(env_parts)}.")

    road_parts = []
    if get_val('roadway_character', None) != None:
        road_parts.append(f"The roadway was a {get_val('roadway_character').lower()}")
    if get_val('trafficway_description', None) != None:
        road_parts.append(f"described as a {get_val('trafficway_description').lower()}")

    posted_speed = get_val('posted_speed_limit', None, lambda x: int(x))
    if posted_speed is not None:
        road_parts.append(f"with a posted speed limit of {posted_speed} mph")

    if get_val('traffic_control_device', None) != None:
        road_parts.append(f"and controlled by a {get_val('traffic_control_device').lower()}")
    
    intersection_related_val = get_val('intersection_related', None)
    if intersection_related_val is not None:
        road_parts.append(f", and was {'' if str(intersection_related_val).lower() == 'true' else 'not '}intersection-related")
    if road_parts:
        narrative_parts.append(f"Roadway context: {', '.join(road_parts)}.")

    # 3. Crash Type & Severity
    type_desc = get_val('crash_type_description', "an unspecified type")
    collision_type = get_val('collision_type', "unspecified collision")
    crash_severity = get_val('crash_severity', "unknown severity")
    narrative_parts.append(f"It was a {type_desc} collision, specifically a {collision_type}, with a severity of {crash_severity}.")

    # 4. Impact (Fatalities & Injuries)
    fatalities = get_val('number_of_fatalities', 0, lambda x: int(x))
    injuries = get_val('number_of_injuries', 0, lambda x: int(x))
    severe_injuries = get_val('number_of_severe_injuries', 0, lambda x: int(x))
    other_injuries = get_val('number_of_other_injuries', 0, lambda x: int(x))
    
    impact_sentence = f"The crash resulted in {fatalities} fatalities"
    if injuries > 0:
        impact_sentence += f" and {injuries} total injuries"
        if severe_injuries > 0 or other_injuries > 0:
            impact_sentence += f" ({severe_injuries} severe, {other_injuries} other injuries)."
        else:
            impact_sentence += "."
    else:
        impact_sentence += " with no injuries reported."
    narrative_parts.append(impact_sentence)

    # 5. Vehicles & Factors
    num_vehicles = get_val('number_of_vehicles_involved', 0, lambda x: int(x))
    narrative_parts.append(f"{num_vehicles} vehicles were involved.")

    vehicle_info = []
    if get_val('vehicle_type', None) != None:
        vehicle_info.append(f"Vehicle type: {get_val('vehicle_type').lower()}")
    
    if get_val('vehicle_body_type_detailed', None) != None:
        vehicle_info.append(f"Body type: {get_val('vehicle_body_type_detailed').lower()}")
    elif get_val('vehicle_body_type', None) != None:
         vehicle_info.append(f"Body type: {get_val('vehicle_body_type').lower()}") # Fallback if detailed is missing

    if get_val('commercial_vehicle_involved', None) != None and str(get_val('commercial_vehicle_involved')).lower() == 'true':
        vehicle_info.append("A commercial vehicle was involved.")
    if get_val('is_large_truck_involved', None) != None and str(get_val('is_large_truck_involved')).lower() == 'true':
        vehicle_info.append("A large truck was involved.")

    if vehicle_info:
        narrative_parts.append(f"Vehicle details: {'; '.join(vehicle_info)}.")

    action_factor_info = []
    if get_val('pre_crash_action_detailed', None) != None:
        action_factor_info.append(f"Pre-crash action: {get_val('pre_crash_action_detailed').lower()}")
    elif get_val('pre_crash_action', None) != None:
        action_factor_info.append(f"Pre-crash action: {get_val('pre_crash_action').lower()}")

    if get_val('apparent_contributing_factor', None) != None:
        action_factor_info.append(f"Apparent contributing factor: {get_val('apparent_contributing_factor').lower()}")
    
    driver_age = get_val('driver_age_vehicle_1', None, lambda x: int(x))
    if driver_age is not None:
        action_factor_info.append(f"Driver age (Vehicle 1): {driver_age}")

    if action_factor_info:
        narrative_parts.append(f"Contributing context: {'; '.join(action_factor_info)}.")
    
    # 6. Reporting Agency
    if get_val('reporting_agency', None) != None:
        narrative_parts.append(f"Reported by: {get_val('reporting_agency')}.")

    final_narrative = " ".join(narrative_parts).strip()
    return final_narrative


def populate_chromadb(df: pd.DataFrame):
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection_obj = client.get_or_create_collection(name=CHROMA_COLLECTION)

    # Check if ChromaDB is already populated for this collection
    if collection_obj.count() > 0:
        print(f"ChromaDB collection '{collection_obj.name}' already contains {collection_obj.count()} documents.")
        print("Skipping re-population. To re-populate, manually delete the ChromaDB data folder (e.g., './chroma_db_reports').")
        return # Exit if already populated

    print(f"Generating descriptive chunks for {len(df)} crash records and populating ChromaDB...")
    batch_size = 50 # Process in batches to manage memory and API calls efficiently
    
    chunks_to_add = []
    chunk_ids_to_add = []
    metadatas_to_add = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        # Construct a comprehensive narrative string from selected columns
        chunk_text = create_narrative(row, relevant_new_col)

        # Skip if the chunk is essentially empty after processing
        if not chunk_text.strip():
            continue

        # Prepare metadata for ChromaDB (ensure no None values, convert types)
        metadata = {}
        
        crash_case_number = row.get('crash_case_number')
        metadata['crash_case_number'] = str(crash_case_number) if pd.notna(crash_case_number) else 'UNKNOWN_CASE'

        if 'crash_case_year' in row and pd.notna(row['crash_case_year']):
            try:
                metadata['crash_case_year'] = int(row['crash_case_year'])
            except (ValueError, TypeError):
                pass

        county_name = row.get('county_name')
        if pd.notna(county_name):
            metadata['county_name'] = str(county_name)

        city_town_name = row.get('city_town_name')
        if pd.notna(city_town_name):
            metadata['city_town_name'] = str(city_town_name)

        crash_severity_val = row.get('crash_severity')
        if pd.notna(crash_severity_val):
            metadata['crash_severity'] = str(crash_severity_val)
        
        crash_type_desc_val = row.get('crash_type_description')
        if pd.notna(crash_type_desc_val):
            metadata['crash_type_description'] = str(crash_type_desc_val)
        
        chunks_to_add.append(chunk_text)
        chunk_ids_to_add.append(f"crash_{metadata['crash_case_number']}_{idx}")
        metadatas_to_add.append(metadata)

        # Add to ChromaDB in batches
        if len(chunks_to_add) >= batch_size:
            batch_embeddings = []
            valid_batch_chunks = []
            valid_batch_ids = []
            valid_batch_metadatas = []

            for j, chunk in enumerate(chunks_to_add):
                embedding = get_embedding(chunk)
                if embedding:
                    batch_embeddings.append(embedding)
                    valid_batch_chunks.append(chunk)
                    valid_batch_ids.append(chunk_ids_to_add[j])
                    valid_batch_metadatas.append(metadatas_to_add[j])
                else:
                    print(f"Skipping chunk {chunk_ids_to_add[j]} due to embedding error.")

            if valid_batch_ids: # Only add if there are valid items
                collection_obj.add(
                    embeddings=batch_embeddings,
                    documents=valid_batch_chunks,
                    metadatas=valid_batch_metadatas,
                    ids=valid_batch_ids
                )
            # Reset for next batch
            chunks_to_add = []
            chunk_ids_to_add = []
            metadatas_to_add = []
    
    # Add any remaining items after the loop
    if chunks_to_add:
        batch_embeddings = []
        valid_batch_chunks = []
        valid_batch_ids = []
        valid_batch_metadatas = []

        for j, chunk in enumerate(chunks_to_add):
            embedding = get_embedding(chunk)
            if embedding:
                batch_embeddings.append(embedding)
                valid_batch_chunks.append(chunk)
                valid_batch_ids.append(chunk_ids_to_add[j])
                valid_batch_metadatas.append(metadatas_to_add[j])
            else:
                print(f"Skipping chunk {chunk_ids_to_add[j]} due to embedding error.")

        if valid_batch_ids:
            collection_obj.add(
                embeddings=batch_embeddings,
                documents=valid_batch_chunks,
                metadatas=valid_batch_metadatas,
                ids=valid_batch_ids
            )

    print(f"ChromaDB collection '{collection_obj.name}' now has {collection_obj.count()} documents.")

def main():
    print("Preprocessing started")

    # Load CSV into dataframe
    df = load_data(CSV_FILE)
    
    # Convert dataframe to SQL table
    populate_sqldb(df)
    
    # Convert dataframe to Vector database
    populate_chromadb(df)

if __name__ == "__main__":
    main()
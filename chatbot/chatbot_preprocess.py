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
    'CaseNumber', 
    'CaseYear', 'CrashDate', 'CrashTimeF', 
    'CrashType', 'CrashSever','LightCondi', 
    'WeatherCon', 'RoadwayCha', 'RoadSurfac',
    'NumberOfFa', 'NumberOfIn', 'NumberOfSe',
    'NumberOfOt', 'NumberOfVe', 'CountyName',
    'CityTownNa', 'OnStreet', 'ClosestCro',
    'POSTED_SPE', 'VehicleTyp', 'PreCrashAc',
]

relevant_new_col = [
    'crash_case_number', 
    'crash_case_year', 'crash_date','crash_time_formatted', 
    'crash_type_description', 'crash_severity','light_condition', 
    'weather_condition', 'roadway_character', 'road_surface_condition',
    'number_of_fatalities', 'number_of_injuries', 'number_of_severe_injuries',
    'number_of_other_injuries', 'number_of_vehicles_involved', 'county_name',
    'city_town_name', 'on_street_name', 'closest_cross_street',
    'posted_speed_limit', 'vehicle_type', 'pre_crash_action',
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
    
def populate_chromadb(df: pd.DataFrame):
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection_obj = client.get_or_create_collection(name=CHROMA_COLLECTION)

    # Check if ChromaDB is already populated for this collection
    # If you want to force a re-population, you can delete the ChromaDB folder manually
    # or add collection_obj.delete_collection() here before adding new data (use with caution!)
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
        narrative_parts = []
        for col in relevant_new_col:
            value = row.get(col) # Use .get() for safer access
            if pd.notna(value) and str(value).strip() != '': # Check for NaN and empty strings
                # Special formatting for certain fields to make the narrative more readable
                if col == 'crash_date' and pd.api.types.is_datetime64_any_dtype(value):
                    narrative_parts.append(f"Date: {value.strftime('%Y-%m-%d')}")
                elif col == 'crash_case_year':
                    narrative_parts.append(f"Year: {int(value)}")
                elif col == 'number_of_fatalities' and int(value) > 0:
                    narrative_parts.append(f"Fatalities: {int(value)}")
                elif col == 'number_of_injuries' and int(value) > 0:
                    narrative_parts.append(f"Injuries: {int(value)}")
                elif col == 'posted_speed_limit':
                    narrative_parts.append(f"Speed Limit: {int(value)} mph")
                else:
                    # General formatting for other relevant columns
                    narrative_parts.append(f"{col.replace('_', ' ').title()}: {value}")
        
        chunk_text = " | ".join(narrative_parts)
        
        # Add a more direct summary at the beginning if possible
        summary_intro = []
        if 'crash_date' in row and pd.notna(row['crash_date']):
             summary_intro.append(f"On {row['crash_date']}")
        if 'crash_time_formatted' in row and pd.notna(row['crash_time_formatted']):
            summary_intro.append(f"at {row['crash_time_formatted']}")
        if 'city_town_name' in row and pd.notna(row['city_town_name']):
            summary_intro.append(f"in {row['city_town_name']}")
        if 'on_street_name' in row and pd.notna(row['on_street_name']):
            summary_intro.append(f"on {row['on_street_name']}")
        
        if summary_intro:
            chunk_text = f"Crash occurred {' '.join(summary_intro)}. Details: {chunk_text}"

        # Skip if the chunk is essentially empty after processing
        if not chunk_text.strip():
            continue

        # Prepare metadata for ChromaDB (useful for filtering/faceted search later if needed)
        # Always include crash_case_number as the ID is generated from it
        metadata = {}
        
        # crash_case_number (always include, use placeholder if missing for ID)
        crash_case_number = row.get('crash_case_number')
        metadata['crash_case_number'] = str(crash_case_number) if pd.notna(crash_case_number) else 'UNKNOWN_CASE'

        # crash_case_year (integer or None, convert to int only if notna)
        if 'crash_case_year' in row and pd.notna(row['crash_case_year']):
            try:
                metadata['crash_case_year'] = int(row['crash_case_year'])
            except (ValueError, TypeError):
                # Handle cases where year might not be convertible to int
                pass # Skip adding this metadata if it's invalid

        # county_name (string or None, convert to string if notna)
        county_name = row.get('county_name')
        if pd.notna(county_name):
            metadata['county_name'] = str(county_name)

        # city_town_name (string or None, convert to string if notna)
        city_town_name = row.get('city_town_name')
        if pd.notna(city_town_name):
            metadata['city_town_name'] = str(city_town_name)

        # crash_severity (string or None, convert to string if notna)
        crash_severity = row.get('crash_severity')
        if pd.notna(crash_severity):
            metadata['crash_severity'] = str(crash_severity)
        
        chunks_to_add.append(chunk_text)
        # Use crash_case_number as a unique ID if it's always unique and present
        # Otherwise, use a combination or the index for uniqueness.
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
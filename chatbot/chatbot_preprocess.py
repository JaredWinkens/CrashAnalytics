import time
import chromadb.errors
import pandas as pd
import sqlite3
import json
import chromadb
from chromadb.utils import embedding_functions
from tqdm.auto import tqdm
from google import genai
from google.genai import types
import concurrent.futures
import ollama

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
API_KEY = config['general']['api_key']
CHROMADB_PATH = config['paths']['chromadb_path']
DB_FILE = config['paths']['db_file']
DATASET_CONFIG_PATH = config['paths']['dataset_config_path'] 
EMBEDDING_MODEL = config['models']['004']
GEN_MODEL = config['models']['1.5-flash'] 

gemini_client = genai.Client(api_key=API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)

def create_sql_table(df: pd.DataFrame, table_name: str):
    conn = sqlite3.connect(DB_FILE)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Table `{table_name}` created successfully")

def get_embedding(text: str):
    tokens = text.split()
    try:
        # Gemini's embedding model has a context limit per call.
        # Ensure 'text' is not excessively long.
        if not text or len(tokens) > 8000: # Example limit, adjust based on model. max is 8192 tokens for 001
            print(f"Warning: Text too long or empty for embedding. Truncating or skipping: {text[:100]}...")
            text = str(tokens[:8000]) # Truncate if too long
            if not text: return None
        response = gemini_client.models.embed_content(
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

def generate_row_summary(row_data: list, relevant_cols: list, data_desc: str, data_name: str, llm_mode: bool = False):
    result = ""
    if llm_mode:
        summary_prompt = f"""
        Dataset Name: {data_name}
        
        Dataset Description: {data_desc}
        
        Column Names: {relevant_cols}
        
        Row Data: {row_data}

        Task:
        Write a concise and human-readable summary of the data row above, incorporating as much relevant context as possible from the schema.

        Do not inlude any preamble, only output the summary.
        """
        #print(summary_prompt)
        response = ollama.generate(model="llama3", prompt=summary_prompt)
        result = response['response']
    else:
        summary = []
        for i, col in enumerate(relevant_cols): summary.append(f"{col}: {row_data[i]}")
        result = " | ".join(summary)
    return result

def create_chroma_collection(df: pd.DataFrame, collection_name: str, collection_desc: str, metas):

    # Check if the collection exists
    try:
        chroma_client.get_collection(name=collection_name)
        # If it exists, delete it
        chroma_client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    except chromadb.errors.NotFoundError:
        # Collection does not exist, no action needed for deletion
        print(f"Collection '{collection_name}' does not exist, creating a new one.")

    collection_obj = chroma_client.create_collection(name=collection_name)
    print(f"Collection '{collection_name}' created successfully.")

    print(f"Generating descriptive chunks for {len(df)} crash records and populating ChromaDB collection {collection_name}...")
    batch_size = 50 # Process in batches to manage memory and API calls efficiently
    
    chunks_to_add = []
    chunk_ids_to_add = []
    metadatas_to_add = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing records for {collection_name}"):
        # Construct a comprehensive narrative string from selected columns
        chunk_text = generate_row_summary(row.tolist(), df.columns.to_list(), collection_desc, collection_name, llm_mode=True)
        #print(f"CHUNK #{idx}: ", chunk_text)
        #time.sleep(1)
        # Skip if the chunk is essentially empty after processing
        if not chunk_text.strip():
            continue

        # Prepare metadata for ChromaDB (ensure no None values, convert types)
        metadata = {}

        for col in metas: metadata[col] = str(row.get(col))
        
        chunks_to_add.append(chunk_text)
        chunk_ids_to_add.append(f'{idx}')
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
            #time.sleep(30)
    
    # Add any remaining items after the loop
    if chunks_to_add:
        batch_embeddings = []
        valid_batch_chunks = []
        valid_batch_ids = []
        valid_batch_metadatas = []

        for j, chunk in enumerate(chunks_to_add):
            embedding = get_embedding(chunk)
            if embedding:
                #print(f"EMBEDDING #{j}: ", embedding)
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

    file = open(DATASET_CONFIG_PATH, "r")
    data_sources = json.load(file)
    sample_size = 100
    # Process datasources
    for source in data_sources:
        name = source['name']
        desc = source['description']
        meta = source['metadata']
        data = pd.read_csv(source['path'], header=0, parse_dates=source['date_columns'], usecols=source['relevant_columns'])
        data = data.rename(columns=source['column_name_map'])
        data_sampled = data.sample(n=sample_size, random_state=42)

        # Create SQL table from dataframe
        create_sql_table(data, name)
        # Create Chroma collection from dataframe
        #create_chroma_collection(data_sampled, name, desc, meta)


if __name__ == "__main__":
    main()
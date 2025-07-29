import random
from google import genai
from google.genai import types, chats
import sqlite3
import json
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import concurrent.futures
from pydantic import BaseModel
import pandas as pd
import plotly.express as px
import plotly.io as pio
from shapely import Point, points
import chatbot.openstreetmap
import geopandas as gpd
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import plotly.io as pio
import crash_heat_map.map_analyzer as mapanalyzer
from enum import Enum

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
CHROMADB_PATH = config['paths']['chromadb_path']
SQLITE_DATABASE_FILE = config['paths']['db_file']
DATASET_CONFIG_PATH = config['paths']['dataset_config_path'] 
API_KEY = config['general']['api_key']
EMBEDDING_MODEL = config['models']['004']
GEN_MODEL = config['models']['2.5-flash']

gemini_client = genai.Client(api_key=API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

  def __call__(self, input: Documents) -> Embeddings:
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=input,
        config=types.EmbedContentConfig(
          task_type="retrieval_query",
        )
    )

    return response.embeddings[0].values
default_ef = GeminiEmbeddingFunction()

def _get_embedding(text: str):
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
                    task_type="retrieval_query",
                    )
                )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Error generating embedding for text: '{text[:50]}...' - {e}")
            return None
        
def get_sqlite_table_schema(table_name: str) -> str:
    """
    Provides the schema for a specified SQLite database table.

    Args:
        table_name: The name of the table, use 'all' to get the schema for all tables.

    Returns:
        A string representing the schema.
    """
    print("TOOL CALL: get_sqlite_table_schema", "Table Name: ",table_name)
    conn = sqlite3.connect(SQLITE_DATABASE_FILE)
    cursor = conn.cursor()
    
    try:
        if table_name.lower() == "all":
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
            schemas = cursor.fetchall()
            if not schemas:
                return "No tables found in the SQLite database."
            return "\n\n".join([f"Table: {name}\nSchema: {sql}" for name, sql in schemas])
        else:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            if not columns:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                if cursor.fetchone():
                    return f"SQLite Table '{table_name}' exists but has no columns defined."
                else:
                    return f"SQLite Table '{table_name}' not found in the database."
            schema_info = [f"{col[1]} {col[2]}{' PRIMARY KEY' if col[5] else ''}{' NOT NULL' if col[3] else ''}" for col in columns]
            return f"Schema for SQLite table '{table_name}':\n" + ", ".join(schema_info)
    except sqlite3.Error as e:
        return f"Error retrieving schema for SQLite table '{table_name}': {e}"
    finally:
        conn.close()

class SQLiteQueryResult(BaseModel):
    """Represents the result of a SQL query."""
    columns: List[str] = Field(description="List of column names.")
    rows: List[List[Any]] = Field(description="List of rows, where each row is a list of values.")
    message: str = Field(description="A descriptive message about the query execution.")

def execute_read_sqlite_query(sql_query: str) -> SQLiteQueryResult:
    """
    Executes a read-only SQL SELECT, PRAGMA, or EXPLAIN query on the SQLite database.
    Only SELECT, PRAGMA, and EXPLAIN queries are allowed for safety. Call this tool when the user's question requires 
    aggregation (e.g., COUNT, SUM, AVG), filtering by precise values (e.g., year, city, street name), 
    or retrieving specific structured data from the database.

    Args:
        sql_query: A single string argument representing the SQL query.
    
    Returns:
        A SQLiteQueryResult object containing the JSON representation of the query result.
    """
    print("TOOL CALL: execute_read_sqlite_query", "SQL Query: ",sql_query)
    if not sql_query.strip().upper().startswith(("SELECT", "PRAGMA", "EXPLAIN")):
        return SQLiteQueryResult(
            columns=[],
            rows=[],
            message="Error: Only SELECT, PRAGMA, and EXPLAIN queries are allowed for security reasons."
        )

    conn = sqlite3.connect(SQLITE_DATABASE_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        columns = [description[0] for description in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        message = f"Query executed successfully. Returned {len(rows)} rows."
        return SQLiteQueryResult(columns=columns, rows=rows, message=message)
    except sqlite3.Error as e:
        return SQLiteQueryResult(
            columns=[],
            rows=[],
            message=f"Error executing query: {e}"
        )
    finally:
        conn.close()

class ChromaCollectionInfo(BaseModel):
    name: str = Field(description="The name of the ChromaDB collection.")
    id: str = Field(description="The ID of the ChromaDB collection.")
    count: int = Field(description="The number of items in the collection.")

def get_chroma_collection_info(collection_name: str) -> List[ChromaCollectionInfo]:
    """
    Provides information about a ChromaDB collection.
    
    Args:
        collection_name: The name of the collection, use 'all' to get the info for all collections.

    Returns:
        A list of ChromaCollectionInfo objects containing the name, id, 
        and number of items in the collection.
    """
    print("TOOL CALL: get_chroma_collection_info", "Collection Name: ",collection_name)
    if chroma_client is None:
        return []
    if collection_name.lower() == 'all':
        collections = chroma_client.list_collections()
        return [ChromaCollectionInfo(name=c.name, id=str(c.id), count=c.count()) for c in collections]
    else:
        c = chroma_client.get_collection(name=collection_name)
        return [ChromaCollectionInfo(name=c.name, id=str(c.id), count=c.count)]

class ChromaQueryResult(BaseModel):
    """Represents a result from a ChromaDB query."""
    id: str = Field(description="The ID of the retrieved document.")
    document: str = Field(description="The content of the retrieved document.")
    distance: Optional[float] = Field(description="The distance (similarity score) of the document to the query.")
    #metadata: Optional[Dict[str, Any]] = Field(description="Optional metadata associated with the document.")

def search_chroma_documents(
    query_text: str,
    n_results: int = 30,
    collection_name: str = "my_documents"
) -> List[ChromaQueryResult]:
    """Retrieves descriptive textual information from a ChromaDB vector database. Call this tool when the user's question asks for 
    general descriptions, summaries, or insights that are likely to be found in narrative text rather than requiring precise numerical aggregation or filtering.

    Args:
        query_text: A single string argument representing the natural language query for retrieval.
        n_results: The number of documents to return.
        collection_name: The name of the collection to query.

    Returns:
        A list of ChromaQueryResult objects each containing a JSON representation of a chroma document.
    """
    print("TOOL CALL: search_chroma_documents ","Query to be embedded: ",query_text, "Collection Name: ",collection_name)
    if chroma_client is None:
        return [] # Return empty if ChromaDB failed to initialize

    try:
        collection = chroma_client.get_collection(name=collection_name)#, embedding_function=default_ef)
    except Exception as e:
        return [ChromaQueryResult(id="error", document=f"Error accessing collection '{collection_name}': {e}", distance=0.0)]

    try:
        query_embedding = _get_embedding(query_text)
        results = collection.query(
            #query_texts=[query_text],
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'distances']#, 'metadatas']
        )
        
        if not results or not results['documents']:
            return []
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append(ChromaQueryResult(
                id=str(results['ids'][0][i]),
                document=results['documents'][0][i],
                distance=results['distances'][0][i],
                #metadata=results['metadatas'][0][i]
            ))
        return formatted_results
    except Exception as e:
        return [ChromaQueryResult(id="error", document=f"Error performing ChromaDB search: {e}", distance=0.0)]

def create_map(map_type: str, data: gpd.GeoDataFrame, lat, lon, hover_name, hover_data):

    if map_type == "scatter_map":
        fig = px.scatter_map(
                data,
                lat=lat,
                lon=lon,
                center=dict(lat=sum(data[lat].tolist())/len(data[lat].tolist()), 
                            lon=sum(data[lon].tolist())/len(data[lon].tolist())),
                zoom=10,
                hover_name=hover_name,
                hover_data=hover_data,
                map_style="open-street-map",
                #title=plot_title
            )
        return fig
    elif map_type == "density_map":
        fig = px.density_map(
                data,
                lat=lat,
                lon=lon,
                radius=2,
                center=dict(lat=sum(data[lat].tolist())/len(data[lat].tolist()), 
                            lon=sum(data[lon].tolist())/len(data[lon].tolist())),
                zoom=10,
                hover_name=hover_name,
                hover_data=hover_data,
                map_style="open-street-map",
                #title=plot_title
            )
        return fig
    else:
        return None


class PlotlyMapResult(BaseModel):
    """Represents the result of a Plotly map visualization, as a JSON string."""
    analysis: str = Field(description="An analysis of the map")
    message: str = Field(description="A descriptive message about the plot generation.")

def visualize_data(
    table_name: str = 'Crash_Reports',
    select_columns: List[str] = ['Latitude', 'Longitude', 'CaseNumber'],
    locations_list: List[str] = [],
    filter_query: str = "",
    plot_title: str = "Generated Plot",
    map_type: str = "scatter_map"
) -> PlotlyMapResult:
    """Generates a Plotly scatter map by querying data from a database. Use this tool when the user explicitly requests a map, 
    geographical visualization, or a spatial representation of data related to roadway safety. It is ideal for showing incident locations, 
    accident hotspots, or the distribution of various safety metrics across an area.

    Args:
        table_name: The SQL table to query
        select_columns: A list of columns to select, by default always select at least 'Latitude', 'Longitude', & 'CaseNumber'.
        locations_list: A list of locations for filering e.g. ['Buffalo', '9825 Seymour Street, Houghton, NY', 'Utica, NY'].
        filter_query: Any other non-location filters that need to be applied to the query e.g. 'CrashType' = 'COLLISION WITH PEDESTRIAN'. 
        DO NOT include any location related filters such as county, city, town, street, etc.
        plot_title: The title of the density map.
        map_type: The type of map to generate e.g. `scatter_map` or `density_map`, the default is scatter_map.

    Returns:
        A PlotlyMapResult object containing the JSON representation of the plot and a status message.
    """
    conn = sqlite3.connect(SQLITE_DATABASE_FILE)
    print("TOOL CALL: visualize_data ", "MAP TYPE: ", map_type)
    try:
        lat = select_columns[0]
        lon = select_columns[1]
        caseNum = select_columns[2]

        where_clause = f"WHERE {filter_query}" if filter_query else ""
        query = f"SELECT {', '.join(select_columns)} FROM {table_name} {where_clause}"

        filtered_df = pd.read_sql_query(sql=query, con=conn)
        geometry = points(filtered_df[lon], filtered_df[lat])
        crashes_gdf = gpd.GeoDataFrame(filtered_df, geometry=geometry, crs="EPSG:4326")
        
        if crashes_gdf.empty:
            return PlotlyMapResult(
                analysis="",
                message="No data found for plotting.")
        
        if locations_list:
            dataframes = []
            for location in locations_list:
                print(location)
                dataframes.append(chatbot.openstreetmap.get_filtered_data(crashes_gdf, location))
            filtered_data = pd.concat(dataframes, ignore_index=True)
        else:
            return PlotlyMapResult(
                analysis="", 
                message="Cannot plot data unless location(s) are specified")
        
        fig = create_map(map_type=map_type,data=filtered_data, lat=lat, lon=lon, hover_name=caseNum, hover_data=select_columns)
        
        plot_json = fig.to_json()
        global vis_data
        vis_data = pio.from_json(plot_json)
        image_bytes = pio.to_image(fig=vis_data, format='png', scale=1, width=1920, height=1080)
        insights = mapanalyzer.generate_response(image=image_bytes, prompt="Analyze the provided map, keep it brief ~100 tokens")
        
        return PlotlyMapResult(
            analysis=insights,
            message=f"Successfully generated a Plotly map titled '{plot_title}'."
        )
    except sqlite3.Error as e:
        return PlotlyMapResult(
            analysis="",
            message=f"Error retrieving data for plot: {e}"
        )
    except Exception as e:
        return PlotlyMapResult(
            analysis="",
            message=f"Error generating Plotly map: {e}"
        )
    finally:
        conn.close()

def create_new_chat_session():
    global chat
    print("New Chat Session Created!")
    
    chat = gemini_client.chats.create(
        model=GEN_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            tools=[
                get_sqlite_table_schema,
                execute_read_sqlite_query,
                get_chroma_collection_info, 
                search_chroma_documents,
                visualize_data
                ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False
            ),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                mode="AUTO", #allowed_function_names=["get_current_temperature"]
                )
            )
        )
    )
    schema = get_sqlite_table_schema('all')
    chat.send_message(f"Here is the schema for the SQL database: \n{schema}")
    collections = get_chroma_collection_info('all')
    chat.send_message(f"Here are the collections in the Chroma database: \n{collections}")

system_prompt = """
You are an interactive roadway safety chatbot designed to provide users with real-time, data-driven insights on roadway safety.

Based on the user's question, determine which tool (or combination of tools) to use.
If a precise count, sum, average, or specific filtered data from the database is needed, use the `execute_read_sqlite_query` tool.
If a general description, summary, or narrative insight is needed, use the `search_chroma_documents` tool.
If the user asks for a map, visualization, or geographical representation of data, use the `visualize_data` tool.
If the question is just for general advice and can be answered without any tools, do so.
If the question falls under mutliple categories, use a combination of the tools above.
If the question is outside the scope of your purpose, state that.
"""

vis_data = None
chat: chats.Chat = None
create_new_chat_session()

def get_agent_response(user_message: str) -> dict[str, any]:

    bot_response_text = ""
    global vis_data
    vis_data = None # Clear previous visulization

    try:
        response = chat.send_message(user_message)
        bot_response_text = response.text

    except Exception as e:
        bot_response_text = f"An error occurred during interaction: {e}"
        print(f"Error in get_agent_response: {e}")
    
    return {"text": bot_response_text, "visualization_data": vis_data}



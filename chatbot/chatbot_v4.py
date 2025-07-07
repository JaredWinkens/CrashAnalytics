import random
from google import genai
from google.genai import types
import sqlite3
import json
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import concurrent.futures
from pydantic import BaseModel
import pandas as pd
import plotly.express as px
import plotly.io as pio
from shapely import Point
import chatbot.openstreetmap
import geopandas as gpd
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import plotly.io as pio

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

# Load crash reports into dataframe
# conn = sqlite3.connect(SQLITE_DATABASE_FILE)
# crash_reports_df = pd.read_sql_query(sql="SELECT * FROM Crash_Reports", con=conn)
# geometry = [Point(xy) for xy in zip(crash_reports_df['Longitude'], crash_reports_df['Latitude'])]
# crashes_gdf = gpd.GeoDataFrame(crash_reports_df, geometry=geometry, crs="EPSG:4326")
# conn.close()

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

  def __call__(self, input: Documents) -> Embeddings:
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=input,
        config=types.EmbedContentConfig(
          task_type="retrieval_document",
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
                    task_type="retrieval_document",
                    )
                )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Error generating embedding for text: '{text[:50]}...' - {e}")
            return None
        
def get_sqlite_table_schema(table_name: str) -> str:
    """
    Provides the schema for a specified SQLite database table.
    Use 'all' to get the schema for all tables.
    """
    conn = sqlite3.connect(SQLITE_DATABASE_FILE)
    cursor = conn.cursor()
    print(table_name)
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
    print(sql_query)
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

def get_chroma_collections_info() -> List[ChromaCollectionInfo]:
    """Provides information about all available ChromaDB collections."""
    if chroma_client is None:
        return []
    collections = chroma_client.list_collections()
    return [ChromaCollectionInfo(name=c.name, id=str(c.id), count=c.count()) for c in collections]

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

class PlotlyDensityMapResult(BaseModel):
    """Represents the result of a Plotly density map visualization, as a JSON string."""
    plot_json: str = Field(description="A JSON string representing the Plotly figure.")
    message: str = Field(description="A descriptive message about the plot generation.")

def visualize_data(
    table_name: str = 'Crash_Reports',
    select_columns: List[str] = ['Latitude', 'Longitude', 'CaseNumber'],
    locations_list: List[str] = [],
    filter_query: str = "",
    plot_title: str = "Generated Plot",
    point_size: int = 1
) -> PlotlyDensityMapResult:
    """Generates a Plotly density map (heatmap) by querying data from a database. Use this tool when the user explicitly requests a map, 
    geographical visualization, or a spatial representation of data related to roadway safety. It is ideal for showing incident locations, 
    accident hotspots, or the distribution of various safety metrics across an area.

    Args:
        table_name: The SQL table to query
        select_columns: A list of columns to select, by default always select 'Latitude', 'Longitude', & 'CaseNumber'
        locations_list: A list of locations for filering e.g. ['Buffalo', 'Houghton, NY', '9825 Seymour Street']
        filter_query: Any other non-location filters that need to be applied to the query e.g. 'CrashType' = 'COLLISION WITH PEDESTRIAN'
        plot_title: The title of the density map.
        point_size: The radius of the density points in the map.

    Returns:
        A PlotlyDensityMapResult object containing the JSON representation of the plot and a status message.
    """
    conn = sqlite3.connect(SQLITE_DATABASE_FILE)

    try:
        lat = select_columns[0]
        lon = select_columns[1]
        caseNum = select_columns[2]

        where_clause = f"WHERE {filter_query}" if filter_query else ""
        query = f"SELECT {', '.join(select_columns)} FROM {table_name} {where_clause}"
        print(query)

        filtered_df = pd.read_sql_query(sql=query, con=conn)
        geometry = [Point(xy) for xy in zip(filtered_df[lon], filtered_df[lat])]
        crashes_gdf = gpd.GeoDataFrame(filtered_df, geometry=geometry, crs="EPSG:4326")
        
        if crashes_gdf.empty:
            return PlotlyDensityMapResult(plot_json="{}", message="No data found for plotting.")
        
        if locations_list:
            dataframes = []
            for location in locations_list:
                print(location)
                dataframes.append(chatbot.openstreetmap.get_filtered_data(crashes_gdf, location))
            filtered_data = pd.concat(dataframes, ignore_index=True)
        else:
            return PlotlyDensityMapResult(plot_json="{}", message="Cannot plot data unless location(s) are specified")
        
        fig = px.density_map(
            filtered_data,
            lat=lat,
            lon=lon,
            radius=point_size,
            center=dict(lat=sum(filtered_data[lat].tolist())/len(filtered_data[lat].tolist()), 
                        lon=sum(filtered_data[lon].tolist())/len(filtered_data[lon].tolist())),
            zoom=10,
            hover_name=caseNum,
            map_style="open-street-map",
            #title=plot_title
        )
        
        plot_json = fig.to_json()
        
        return PlotlyDensityMapResult(
            plot_json=plot_json,
            message=f"Successfully generated a Plotly density map titled '{plot_title}'."
        )
    except sqlite3.Error as e:
        return PlotlyDensityMapResult(
            plot_json="{}",
            message=f"Error retrieving data for plot: {e}"
        )
    except Exception as e:
        return PlotlyDensityMapResult(
            plot_json="{}",
            message=f"Error generating Plotly density map: {e}"
        )
    finally:
        conn.close()

system_prompt = """
You are an interactive roadway safety chatbot designed to provide users with real-time, data-driven insights on roadway safety.

Based on the user's question, determine which tool (or combination of tools) to use.
If a precise count, sum, average, or specific filtered data from the database is needed, use the `execute_read_sqlite_query` tool.
If a general description, summary, or narrative insight is needed, use the `search_chroma_documents` tool.
If the user asks for a map, visualization, or geographical representation of data, use the `visualize_data` tool.
If the question is just for general advice and can be answered without any tools, do so.
If the question is outside the scope of your purpose, state that.
"""
tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
        mode="AUTO", #allowed_function_names=["get_current_temperature"]
    )
)

chat = gemini_client.chats.create(
    model=GEN_MODEL,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0,
        tools=[execute_read_sqlite_query, 
               search_chroma_documents,
               visualize_data],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        ),
        tool_config=tool_config
    )
)

# Send resources to chat
sql_schema = get_sqlite_table_schema('all')
chat.send_message(f"Here is the schema of the SQL Database\n\n{sql_schema}")
collections_info = get_chroma_collections_info()
chat.send_message(f"Here is are all the collections in the ChromaDB Database\n\n{str(collections_info)}")

def get_agent_response(user_message: str) -> dict[str, any]:

    bot_response_text = ""
    visualization_data = None

    
    response = chat.send_message(user_message)

    try:
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_call = part.function_call
                    tool_name = tool_call.name
                    tool_args = tool_call.args

                    print(f"Gemini decided to call tool: {tool_name} with arguments: {tool_args}")

                    # Execute the tool based on Gemini's request
                    if tool_name == "execute_read_sqlite_query":
                        tool_result: SQLiteQueryResult = execute_read_sqlite_query(**tool_args)
                        print(f"SQL Tool Result:\n{str(tool_result)}")
                    
                    elif tool_name == "search_chroma_documents":
                        tool_result: List[ChromaQueryResult] = search_chroma_documents(**tool_args)
                        print(f"RAG Tool Result:\n{str(tool_result)}")

                    elif tool_name == "visualize_data":
                        tool_result: PlotlyDensityMapResult = visualize_data(**tool_args)
                        print(f"VIS Tool Result:")
                        #fig = json.loads(tool_result.plot_json)
                        visualization_data = pio.from_json(tool_result.plot_json)
                        bot_response_text = tool_result.message
                        break
                    
                    else:
                        tool_result = f"Unknown tool requested by Gemini: {tool_name}"
                    
                    # if tool_name != "visualize_map_data":
                    #     chat.send_message(str(tool_result))

                    follow_up_response = chat.send_message(f"Tool output for {tool_name}:\n{str(tool_result)}")
                    if follow_up_response.candidates and follow_up_response.candidates[0].content.parts:
                        for follow_up_part in follow_up_response.candidates[0].content.parts:
                            if follow_up_part.text:
                                bot_response_text += "\n" + follow_up_part.text
                elif part.text:
                    bot_response_text += part.text
        else:
            bot_response_text = "Gemini did not provide a text response or tool call."

    except Exception as e:
        bot_response_text = f"An error occurred during interaction: {e}"
        print(f"Error in get_agent_response: {e}")

    return {"text": bot_response_text, "visualization_data": visualization_data}


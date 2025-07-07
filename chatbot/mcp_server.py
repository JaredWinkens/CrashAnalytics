import sqlite3
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import json
from google.genai import types
from google import genai
import plotly.express as px # Import plotly express
import pandas as pd # Import pandas for data handling

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
CHROMADB_PATH = config['paths']['chromadb_path']
SQLITE_DATABASE_FILE = config['paths']['db_file']
API_KEY = config['general']['api_key']
EMBEDDING_MODEL = config['models']['004']

gemini_client = genai.Client(api_key=API_KEY)

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

  def __call__(self, input: Documents) -> Embeddings:
    title = "Custom query"
    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=input,
        config=types.EmbedContentConfig(
          task_type="retrieval_document",
          title=title
        )
    )

    return response.embeddings[0].values
  
# Initialize the FastMCP server
mcp = FastMCP("Multi-Database Explorer")

def initalize_sqlite_database():
    pass

initalize_sqlite_database()

chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)

default_ef = GeminiEmbeddingFunction()

def initalize_chroma_database():
   pass

initalize_chroma_database()

@mcp.resource("sqlite://schema/{table_name}", title="SQLite Table Schema")
def get_sqlite_table_schema(table_name: str) -> str:
    """
    Provides the schema for a specified SQLite database table.
    Use 'sqlite://schema/all' to get the schema for all tables.
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

@mcp.tool(title="Execute Read-Only SQLite Query")
def execute_read_sqlite_query(sql_query: str) -> SQLiteQueryResult:
    """
    Executes a read-only SQL SELECT, PRAGMA, or EXPLAIN query on the SQLite database.
    Only SELECT, PRAGMA, and EXPLAIN queries are allowed for safety.
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

@mcp.resource("chromadb://collections/all", title="ChromaDB Collections Info")
def get_chroma_collections_info() -> List[ChromaCollectionInfo]:
    """Provides information about all available ChromaDB collections."""
    if chroma_client is None:
        return []
    collections = chroma_client.list_collections()
    return [ChromaCollectionInfo(name=c.name, id=c.id, count=c.count()) for c in collections]

class ChromaQueryResult(BaseModel):
    """Represents a result from a ChromaDB query."""
    id: str = Field(description="The ID of the retrieved document.")
    document: str = Field(description="The content of the retrieved document.")
    distance: Optional[float] = Field(description="The distance (similarity score) of the document to the query.")
    metadata: Optional[Dict[str, Any]] = Field(description="Optional metadata associated with the document.")

@mcp.tool(title="Search ChromaDB Documents")
def search_chroma_documents(
    query_text: str,
    n_results: int = 5,
    collection_name: str = "my_documents"
) -> List[ChromaQueryResult]:
    """
    Performs a vector similarity search in a ChromaDB collection.
    By default, searches the 'my_documents' collection.
    """
    if chroma_client is None:
        return [] # Return empty if ChromaDB failed to initialize

    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=default_ef)
    except Exception as e:
        return [ChromaQueryResult(id="error", document=f"Error accessing collection '{collection_name}': {e}", distance=0.0)]

    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'distances', 'metadatas']
        )
        
        if not results or not results['documents']:
            return []

        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append(ChromaQueryResult(
                id=results['ids'][0][i],
                document=results['documents'][0][i],
                distance=results['distances'][0][i],
                metadata=results['metadatas'][0][i]
            ))
        return formatted_results
    except Exception as e:
        return [ChromaQueryResult(id="error", document=f"Error performing ChromaDB search: {e}", distance=0.0)]

class PlotlyDensityMapResult(BaseModel):
    """Represents the result of a Plotly density map visualization, as a JSON string."""
    plot_json: str = Field(description="A JSON string representing the Plotly figure.")
    message: str = Field(description="A descriptive message about the plot generation.")

@mcp.tool(title="Visualize Data as Plotly Density Map")
def visualize_data(
    table_name: str = 'Crash_Reports',
    latitude_column: str = 'Latitude',
    longitude_column: str = 'Longitude',
    filter_query: Optional[str] = "",
    plot_title: str = "Generated Plot",
    point_size: int = 10
) -> PlotlyDensityMapResult:
    """Generates a Plotly density map (heatmap) by querying data from a database.

    Args:
        table_name: The SQL table to query
        latitude_column: The name of the latitude column in the table
        longitude_column: The name of the longitude column in the table
        filter_query: Any filters that need to be applied to the query e.g. `'CityTownName' = 'Buffalo'`
        plot_title: The title of the density map.
        point_size: The radius of the density points in the map.

    Returns:
        A PlotlyDensityMapResult object containing the JSON representation of the plot and a status message.
    """
    conn = sqlite3.connect(SQLITE_DATABASE_FILE)
    cursor = conn.cursor()

    try:
        select_cols = [latitude_column, longitude_column]
            
        where_clause = f"WHERE {filter_query}" if filter_query else ""
        query = f"SELECT {', '.join(select_cols)} FROM {table_name} {where_clause}"
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return PlotlyDensityMapResult(plot_json="{}", message="No data found for plotting.")
        
        df_columns = ['lat', 'lon']
        
        df = pd.DataFrame(rows, columns=df_columns)
        
        fig = px.density_mapbox(
            df,
            lat='lat',
            lon='lon',
            radius=point_size,
            center=dict(lat=sum(df['lat'].tolist())/len(df['lat'].tolist()), lon=sum(df['lon'].tolist())/len(df['lon'].tolist())),
            zoom=1,
            mapbox_style="open-street-map",
            title=plot_title
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

if __name__ == "__main__":
    print(f"MCP server '{mcp.name}' is starting...")
    mcp.run()
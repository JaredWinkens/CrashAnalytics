from google import genai
from google.genai import types
import sqlite3
import json
import chromadb
import concurrent.futures
from pydantic import BaseModel
import pandas as pd
import plotly.express as px

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
API_KEY = config['general']['api_key']
CHROMADB_PATH = config['paths']['chromadb_path']
DB_FILE = config['paths']['db_file']
DATASET_CONFIG_PATH = config['paths']['dataset_config_path'] 
EMBEDDING_MODEL = config['models']['004'] 
GEN_MODEL = config['models']['2.0-flash']

gemini_client = genai.Client(api_key=API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
chroma_collections = chroma_client.list_collections()
collections_dict = {}
for collection in chroma_collections:
    collections_dict[collection.name] = collection

def call_with_timeout(func, timeout, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Function '{func.__name__}' timed out after {timeout} seconds.")
            return f"Function '{func.__name__}' timed out after {timeout} seconds." # Or raise a custom exception
        except Exception as e:
            print(f"Function '{func.__name__}' raised an exception: {e}")
            return f"Function '{func.__name__}' raised an exception: {e}"# Re-raise the original exception

def get_table_schema_dict(db_path: str, table_name: str, table_desc: str, include_samples: bool = True, sample_limit: int = 1) -> dict:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get column metadata
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)

    # Sample data (optional)
    sample_row = {}
    if include_samples:
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {sample_limit};")
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            sample_row = {col: [] for col in col_names}

            for row in rows:
                for col_name, value in zip(col_names, row):
                    sample_row[col_name].append(str(value)) if value is not None else None
                    #print("COL NAME: ",col_name, "COL VAL: ", value)
        except Exception as e:
            print(f"[!] Failed to fetch sample data from '{table_name}': {e}")

    conn.close()

    # Build dict structure
    schema = {
        "table_name": table_name,
        "table_desc": table_desc,
        "columns": []
    }

    for col in columns:
        col_name = col[1]
        col_type = col[2]
        col_sample = sample_row.get(col_name) if include_samples else None
        schema["columns"].append({
            "name": col_name,
            "type": col_type,
            "sample": col_sample
        })

    return schema

def format_schema(schema_dict: dict) -> str:
    """Formats schema dictionary into a readable prompt-friendly string."""
    lines = [f"Table Name: {schema_dict['table_name']}", f"Table Description: {schema_dict['table_desc']}", "Columns:"]
    for col in schema_dict["columns"]:
        sample = f" e.g., {', '.join(str(s) for s in col['sample'])}" if col.get("sample") else ""
        lines.append(f'- {col["name"]} ({col["type"]}){sample}')
    return "\n".join(lines)

file = open(DATASET_CONFIG_PATH, "r")
data_sources = json.load(file)
table_schemas = []
for source in data_sources:
    schema = get_table_schema_dict(DB_FILE, source['name'], source['description'], True, 3)
    formatted_schema = format_schema(schema)
    table_schemas.append(formatted_schema)

DB_SCHEMA_DESCRIPTION = "\n\n".join(table_schemas)

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

def execute_sql_query(query: str):
    """
    Executes a SQL query against the SQLite database and returns results.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return {"status": "success", "data": results}
    except sqlite3.Error as e:
        if conn: conn.close()
        return {"status": "error", "message": f"SQLite error: {e}"}
    except Exception as e:
        if conn: conn.close()
        return {"status": "error", "message": f"General error: {e}"}
    
# Describe the SQL tool for Gemini's understanding
SQL_TOOL_DESCRIPTION = f"""
Tool Name: `execute_sql_query`
Description: Executes a SQL query against the SQLite database.
Usage: Call this tool when the user's question requires aggregation (e.g., COUNT, SUM, AVG), filtering by precise values (e.g., year, city, street name), or retrieving specific structured data from the database.
Input: A single string argument representing the SQL query.
Database Schema:
{DB_SCHEMA_DESCRIPTION}
Example SQL queries:
- How many crashes in each year?: `SELECT Case_Year, COUNT(Case_Number) AS Total_Crashes FROM Intersection GROUP BY Case_Year ORDER BY Case_Year DESC;`
- What is the average number of injuires for crashes involving a commercial vehicle?: `SELECT AVG(Number_of_Injuries) AS Average_Injuries FROM Intersection WHERE Commercial_Vehicle_Involved = 1;`
- Retrive the age and sex of people involved in crashes resulting in a serious injury: `SELECT ip.Case_Number, ip.Person_Type, ip.Age, ip.Sex FROM IntersectionPerson ip WHERE ip.Injury_Severity = 'A - SERIOUS INJURY';`

Important:
- Be precise with column names as listed in the schema
- Join tables when necessary
- Only ouput one SQL query
- If the query cannot be answered by SQL, do not generate a SQL query.
"""

def retrieve_relevant_chunks(query: str, top_k: int = 30):
    """
    Retrieves relevant textual chunks from ChromaDB.
    """
    collections_list = []
    collection_desc = [source['description'] for source in data_sources]
    collection_names = [source['name'] for source in data_sources]
    for i in range(len(collection_names)): collections_list.append(f'- Collection Name: {collection_names[i]}, Collection Description: {collection_desc[i]}')
    collections_str = "\n".join(collections_list)

    # Prompt the LLM to identify relevant collections
    prompt = (f"The user is asking: '{query}'\n"
              f"You have the following ChromaDB collections available, which contain crash data organized by different criteria: \n{collections_str}.\n"
              "Based on the user's query, list ONLY the names of the most relevant collections (comma-separated, no extra text). "
              "If no specific collection seems relevant, just say 'all'."
              "Example: crashes_2020, pedestrian_crashes")
    
    # This is a simplified call; in a real agent, you'd use a more robust LLM interaction
    response_from_llm = gemini_client.models.generate_content(model=GEN_MODEL, contents=prompt).text.strip()

    if "all" in response_from_llm or not response_from_llm:
        relevant_collections_to_query = collections_dict.values()
        print("Agent decided to query all collections.")
    else:
        relevant_names = [name.strip() for name in response_from_llm.split(',')]
        relevant_collections_to_query = [collections_dict[name] for name in relevant_names if name in collections_dict]
        print(f"Agent decided to query specific collections: {', '.join(relevant_names)}")
        if not relevant_collections_to_query:
            print("No matching collections found based on agent's decision, querying all as fallback.")
            relevant_collections_to_query = collections_dict.values() # Fallback
    
    combined_results = []
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return {"status": "error", "message": "Could not generate embedding for query."}
    
    for collection_obj in relevant_collections_to_query:
        try:
            results = collection_obj.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'distances', 'metadatas']
            )
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    combined_results.append({
                        'collection': collection_obj.name,
                        'document': results['documents'][0][i],
                        'distance': results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
        except Exception as e:
            print(f"Error querying {collection_obj.name}: {e}")
            continue
    
    combined_results.sort(key=lambda x: x['distance'])
    context_docs = []
    for item in combined_results[:top_k]:
        context_docs.append(f"Source (Collection: {item['collection']}):\n{item['document']}")
    context = "\n\n".join(context_docs)
    #relevant_docs = [result['document'] for result in combined_results]
    return {"status": "success", "data": context}
    
# Describe the RAG tool for Gemini's understanding
RAG_TOOL_DESCRIPTION = f"""
Tool Name: `retrieve_relevant_chunks`
Description: Retrieves descriptive textual information from a ChromaDB vector database.
Usage: Call this tool when the user's question asks for general descriptions, summaries, or insights that are likely to be found in narrative text rather than requiring precise numerical aggregation or filtering.
Input: A single string argument representing the natural language query for retrieval.
Example usage:
- "Describe typical crash scenarios."
- "What generally causes crashes on highways?"
"""

def visualize_data(sql_query: str):
    fig = None
    try:
        if 'Longitude' in sql_query and 'Latitude' in sql_query and 'Case_Number' in sql_query:
            data = execute_sql_query(sql_query)['data']
            filtered_data = pd.DataFrame(data)
            fig = px.scatter_map(
                filtered_data,
                lat="Latitude",
                lon="Longitude",
                color="Injuries",
                size="Injuries",
                size_max=10, # Max size of markers
                zoom=10, # Initial zoom level
                hover_name="Case_Number",
                hover_data={
                "Case_Number": True,
                "Crash_Severity": True,
                "Crash_Date": True,
                "Crash_Time_Formatted": True,
                "Crash_Type": True,
                "City_Town_Name": True,
                "On_Street_Name": True
                },
                map_style="open-street-map"
            )
            fig.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                mapbox_bounds={"west": filtered_data['Longitude'].min() - 0.01,
                            "east": filtered_data['Longitude'].max() + 0.01,
                            "south": filtered_data['Latitude'].min() - 0.01,
                            "north": filtered_data['Latitude'].max() + 0.01},
                # Alternatively, set initial center and zoom
                # mapbox_center={"lat": filtered_df['latitude'].mean(), "lon": filtered_df['longitude'].mean()},
                # mapbox_zoom=10
            )
            return {"status": "success", "data": data}, fig
        else:
            return {"status": "error", "message": "Error: The SQL query did not provide appropriate data for mapping. Please ensure it selects latitude, longitude, and any relevant data points."}, fig
    except Exception as e:
        return {"status": "error", "message": f"An error occurred while generating the map: {str(e)}"}, fig

VISUALIZATION_TOOL_DESCRIPTION = """
Tool Name: `visualize_data`
Description: This tool is designed to visualize roadway safety data on an interactive geographical map using Plotly. It executes a provided SQL query, retrieves relevant data (specifically latitude and longitude, along with any other desired numerical or categorical data for visualization), and then renders it on a map.
Usage: Use this tool when the user explicitly requests a map, geographical visualization, or a spatial representation of data related to roadway safety. It is ideal for showing incident locations, accident hotspots, or the distribution of various safety metrics across an area.
Input: A SQL query as a string. The query must always select at least the following columns

Required Columns to Query:
- Case_Number
- X_Coordinate
- Y_Coordinate
- Crash_Severity
- Crash_Date
- Crash_Time_Formatted
- Crash_Type
- City_Town_Name
- On_Street_Name
- Number_of_Injuries

Example SQL queries:
- Plot all crashes: `SELECT Case_Number, X_Coordinate AS Longitude, Y_Coordinate AS Latitude, Crash_Severity, Crash_Date, Crash_Time_Formatted, Crash_Type, City_Town_Name, On_Street_Name, Number_of_Injuries AS Injuries FROM Intersection;`
- Visualize crashes in 2022 where people were injured: `SELECT Case_Number, X_Coordinate AS Longitude, Y_Coordinate AS Latitude, Crash_Severity, Crash_Date, Crash_Time_Formatted, Crash_Type, City_Town_Name, On_Street_Name, Number_of_Injuries AS Injuries FROM Intersection WHERE Case_Year = 2022 AND Crash_Severity = 'INJURY';`
- Show me crashes involving pedestrians, order by date: `SELECT Case_Number, X_Coordinate AS Longitude, Y_Coordinate AS Latitude, Crash_Severity, Crash_Date, Crash_Time_Formatted, Crash_Type, City_Town_Name, On_Street_Name, Number_of_Injuries AS Injuries FROM Intersection WHERE Crash_Type = 'COLLISION WITH PEDESTRIAN' ORDER BY Crash_Date DESC;`
"""

def get_sql_tool_response(sql_query: str, user_query: str):
    print(f"Executing SQL Query: {sql_query}")
    tool_result = call_with_timeout(execute_sql_query, 30, sql_query)

    if tool_result['status'] == "success":
        print(f"SQL Tool Result: {tool_result['data']}")
        summary_prompt = f"""
        The user asked: "{user_query}"
        I executed a SQL query and got the following result:
        {json.dumps(tool_result['data'], indent=2)}

        Please provide a concise and clear answer to the user's question based on these results.
        If query resulted in an error, empty data or the answer is not in the information, state that.
        """
        final_answer_response = gemini_client.models.generate_content(
                                                model=GEN_MODEL,
                                                contents=summary_prompt,
                                            ).text
    else: #tool_result['status'] == "error"
        print(f"SQL Tool Result: {tool_result['message']}")
        final_answer_response = tool_result['message']

    return BotResponse(text=final_answer_response)

def get_rag_tool_response(rag_query: str, user_query: str):    
    print(f"Executing RAG Query for: {rag_query}")
    tool_result = call_with_timeout(retrieve_relevant_chunks, 30, rag_query)

    if tool_result['status'] == "success":
        print(f"RAG Tool Result: {tool_result['data']}")
        final_answer_prompt = f"""
        The user asked: "{user_query}"
        I retrieved the following relevant information:
        {tool_result['data']}

        Please answer the user's question based ONLY on the provided information.
        If query resulted in an error, empty data or the answer is not in the information, state that.
        """
        final_answer_response = gemini_client.models.generate_content(
                                            model=GEN_MODEL,
                                            contents=final_answer_prompt,
                                        ).text
    else: #tool_result['status'] == "error"
        print(f"RAG Tool Result: {tool_result['message']}")
        final_answer_response = tool_result['message']

    return BotResponse(text=final_answer_response)

def get_visualization_tool_response(sql_query: str, user_query: str):
    print(f"Executing SQL Query for: {sql_query}")
    tool_result, fig = call_with_timeout(visualize_data, 30, sql_query)
    
    if tool_result['status'] == "success": 
        print(f"VIS Tool Result: {tool_result['data']}")
        final_answer_prompt = f"""
        The user asked: "{user_query}"
        I executed a SQL query and got the following data points:
        {json.dumps(tool_result['data'], indent=2)}

        Please analyze the data points and draw helpful insights such as trends and hotspots in one paragraph.
        If query resulted in an error, empty data or the answer is not in the information, state that.
        """
        final_answer_response = gemini_client.models.generate_content(
                                            model=GEN_MODEL,
                                            contents=final_answer_prompt,
                                        ).text
    else: #tool_result['status'] == "error"
        print(f"VIS Tool Result: {tool_result['message']}")
        final_answer_response = tool_result['message']

    return BotResponse(text=final_answer_response, map=fig)

class Tool(BaseModel):
    function_name: str
    function_arguments: list[str]

class BotResponse():
    def __init__(self, text="", map=None, img=None):
        self.text=text
        self.map=map
        self.img=img

def get_agent_response(user_query: str) -> BotResponse:
    """
    Gemini decides which tool to use (SQL or RAG) and executes it.
    """
    # Prompt Gemini to decide which tool to use
    decision_prompt = f"""
    You are an interactive roadway safety chatbot designed to provide users with real-time, data-driven insights on roadway safety. 
    You have access to three tools:

    {SQL_TOOL_DESCRIPTION}

    {RAG_TOOL_DESCRIPTION}

    {VISUALIZATION_TOOL_DESCRIPTION}

    Based on the user's question, determine which tool (or combination of tools) to use and how to call it.
    If a precise count, sum, average, or specific filtered data from the database is needed, use the `execute_sql_query` tool.
    If a general description, summary, or narrative insight is needed, use the `retrieve_relevant_chunks` tool.
    If the user asks for a map, visualization, or geographical representation of data, use the `visualize_data` tool.
    If the question is just for general advice and can be answered without any tools, do so.
    If the question is outside the scope of your purpose, state that.

    Output Format:
    - If using the `execute_sql_query` tool: `function_name` = execute_sql_query, `function_arguments` = ['YOUR_SQL_QUERY_HERE;']
    - If using the `retrieve_relevant_chunks` tool: `function_name` = retrieve_relevant_chunks, `function_arguments` = ['YOUR_NATURAL_LANGUAGE_QUERY_HERE']
    - If using the `visualize_data` tool: `function_name` = visualize_data, `function_arguments` = ['YOUR_SQL_QUERY_HERE;']
    - If within scope and no tool needed: `function_name` = no_tool_needed, `function_arguments` = ['YOUR_RESPONSE_HERE']
    - If outside the scope of your purpose: `function_name` = invalid_query, `function_arguments` = ['YOUR_RESPONSE_HERE']

    User Question: "{user_query}"
    """
    print(f"\n--- Gemini's Tool Decision Prompt ---\n{decision_prompt}\n----------------------------------")
    try:
        response = gemini_client.models.generate_content(
            model=GEN_MODEL,
            contents=decision_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[Tool]
            )
        )
        tools_to_use: list[Tool] = response.parsed
        tool_responses = []
        for tool in tools_to_use:
            print(f"Gemini's decision: {tool.function_name}\n Args: {tool.function_arguments}")
            if tool.function_name == 'execute_sql_query':
                sql_query = tool.function_arguments[0].strip("'\"") # Remove potential quotes around the query
                tool_responses.append({"Tool_Name": tool.function_name, "Result": get_sql_tool_response(sql_query, user_query)})

            elif tool.function_name == 'retrieve_relevant_chunks':
                rag_query = tool.function_arguments[0].strip("'\"") # Remove potential quotes
                tool_responses.append({"Tool_Name": tool.function_name, "Result": get_rag_tool_response(rag_query, user_query)})
            
            elif tool.function_name == 'visualize_data':
                sql_query = tool.function_arguments[0].strip("'\"")
                tool_responses.append({"Tool_Name": tool.function_name, "Result": get_visualization_tool_response(sql_query, user_query)})
            
            elif tool.function_name == 'no_tool_needed': 
                tool_responses.append({"Tool_Name": tool.function_name, "Result": BotResponse(text=tool.function_arguments[0].strip("'\""))})
            
            elif tool.function_name == 'invalid_query':
                tool_responses.append({"Tool_Name": tool.function_name, "Result": BotResponse(text=tool.function_arguments[0].strip("'\""))})
            
            else:
                tool_responses.append({"Tool Name": tool.function_name, "Result": BotResponse(text="Error: Unknown tool function identified by Gemini.")})
        if len(tool_responses) == 1:
            return tool_responses[0]['Result']
        else:
            tool_responses_list = []
            for response in tool_responses:
                tool_responses_list.append(f"Tool Name: {response['Tool_Name']}, Tool Result: {response['Result'].text}")
            tool_responses_list_str = "\n\n".join(tool_responses_list)
            final_answer_prompt = f"""
            The user asked: "{user_query}"
            I retived the following information from the RAG tools below.

            {tool_responses_list_str}

            Please summarize the results of the tools to answer the user's question.
            """
            response = gemini_client.models.generate_content(model=GEN_MODEL, contents=final_answer_prompt)
            return BotResponse(text=response.text)
    except Exception as e:
        return BotResponse(text=f"An error occurred while retriving the data: {e}")

if __name__ == "__main__":
    print("\n--- Hybrid Agentic System Ready ---")
    print("Type your questions about NY crash reports. Type 'exit' to quit.")

    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'exit':
            break

        answer = get_agent_response(user_question)
        print("\n--- Gemini's Final Answer ---")
        print(answer)
        print("----------------------------")
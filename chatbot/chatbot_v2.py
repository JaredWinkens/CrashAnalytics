from google import genai
from google.genai import types
import sqlite3
import json
import chromadb
import concurrent.futures
from pydantic import BaseModel
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
- Show me all crashes in Buffalo where people were injured and the road was wet: `SELECT Case_Number, Crash_Date, Crash_Time_Formatted, Maximum_Injury_Severity FROM Intersection WHERE City_Town_Name = 'Buffalo' AND Road_Surface_Condition = 'WET' AND Crash_Severity = 'INJURY' ORDER BY Crash_Date DESC;`
- Retrive the age and sex of people involved in crashes resulting in a serious injury: `SELECT ip.Case_Number, ip.Person_Type, ip.Age, ip.Sex FROM IntersectionPerson ip WHERE ip.Injury_Severity = 'A - SERIOUS INJURY';`
- Show me the collision type and max injury severity for each person involved in an accident in Erie county: `SELECT i.Case_Number, i.Collision_Type, i.Crash_Date, ip.Person_Type, ip.Injury_Severity AS Person_Injury_Severity FROM Intersection i JOIN IntersectionPerson ip ON i.Case_Number = ip.Case_Number WHERE i.County_Name = 'Erie' ORDER BY Crash_Date DESC, i.Case_Number, ip.Person_Sequence_Number;`

Important:
- Be precise with column names as listed in the schema
- Join tables when necessary
- Only ouput one SQL query
- If the query cannot be answered by SQL, do not generate a SQL query.
"""

def retrieve_relevant_chunks(query: str, top_k: int = 10):
    """
    Retrieves relevant textual chunks from ChromaDB.
    """
    combined_results = []
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return {"status": "error", "message": "Could not generate embedding for query."}
    
    for collection_obj in chroma_collections:
        try:
            results = collection_obj.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'distances']
            )
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    combined_results.append({
                        'collection': collection_obj.name,
                        'document': results['documents'][0][i],
                        'distance': results['distances'][0][i]
                    })
        except Exception as e:
            print(f"Error querying {collection_obj.name}: {e}")
            continue
    
    combined_results.sort(key=lambda x: x['distance'])
    relevant_docs = []
    for result in combined_results: relevant_docs.append(result['document']) 
    return {"status": "success", "data": relevant_docs}
    
# Describe the RAG tool for Gemini's understanding
RAG_TOOL_DESCRIPTION = f"""
Tool Name: `retrieve_relevant_chunks`
Description: Retrieves descriptive textual information from a knowledge base of crash reports.
Usage: Call this tool when the user's question asks for general descriptions, summaries, or insights that are likely to be found in narrative text rather than requiring precise numerical aggregation or filtering.
Input: A single string argument representing the natural language query for retrieval.
Example usage:
- "Describe typical crash scenarios."
- "What generally causes crashes on highways?"
"""

def get_sql_tool_response(sql_query: str, user_query: str):
    print(f"Executing SQL Query: {sql_query}")
    tool_result = call_with_timeout(execute_sql_query, 30, sql_query)
    print(f"SQL Tool Result: {tool_result}")

    # Now, ask Gemini to summarize the SQL result
    summary_prompt = f"""
    The user asked: "{user_query}"
    I executed a SQL query and got the following result:
    {json.dumps(tool_result['data'], indent=2)}

    Please provide a concise and clear answer to the user's question based on these results.
    If the query resulted in an error or empty data, state that no information was found.
    """
    final_answer_response = gemini_client.models.generate_content(
                                            model=GEN_MODEL,
                                            contents=summary_prompt,
                                        )
    return final_answer_response.text

def get_rag_tool_response(rag_query: str, user_query: str):    
    print(f"Executing RAG Query for: {rag_query}")
    tool_result = call_with_timeout(retrieve_relevant_chunks, 30, rag_query)
    print(f"RAG Tool Result: {tool_result}")

    if tool_result['status'] == 'success' and tool_result['data']:
        context = "\n\n".join(tool_result['data'])
        final_answer_prompt = f"""
        The user asked: "{user_query}"
        I retrieved the following relevant information:
        {context}

        Please answer the user's question based ONLY on the provided information.
        If the answer is not in the information, state that.
        """
        final_answer_response = gemini_client.models.generate_content(
                                            model=GEN_MODEL,
                                            contents=final_answer_prompt,
                                        )
        return final_answer_response.text
    else:
        return f"I couldn't find relevant information for your question via RAG: {tool_result.get('message', '')}"

def get_sql_rag_tool_response(sql_query: str, rag_query: str, user_query: str):
    
    print(f"Executing SQL Query: {sql_query}")
    sql_result = call_with_timeout(execute_sql_query, 30, sql_query)
    print(f"SQL Tool Result: {sql_result}")

    print(f"Executing RAG Query for: {rag_query}")
    rag_result = call_with_timeout(retrieve_relevant_chunks, 30, rag_query)
    print(f"RAG Tool Result: {rag_result}")

    if rag_result['status'] == 'success' and rag_result['data']:
        rag_context = "\n\n".join(rag_result['data'])
        final_answer_prompt = f"""
        The user asked: "{user_query}"
        I retrieved the following relevant information:

        SQL Query Result:
        {json.dumps(sql_result['data'], indent=2)}

        RAG Query Result:
        {rag_context}

        Please answer the user's question based on the provided information.
        If the answer is not in the information, state that.
        """
        final_answer_response = gemini_client.models.generate_content(
                                    model=GEN_MODEL,
                                    contents=final_answer_prompt,
                                )
        return final_answer_response.text
    else:
        return f"I couldn't find relevant information for your question via RAG: {rag_result.get('message', '')}"

class Tool(BaseModel):
    function_name: str
    function_arguments: list[str]

def get_agent_response(user_query: str):
    """
    Gemini decides which tool to use (SQL or RAG) and executes it.
    """
    # Prompt Gemini to decide which tool to use
    decision_prompt = f"""
    You are an intelligent assistant that can answer questions about NY crash reports.
    You have access to two tools:

    {SQL_TOOL_DESCRIPTION}

    {RAG_TOOL_DESCRIPTION}

    Based on the user's question, determine which tool (or combination of tools) to use and how to call it.
    If a precise count, sum, average, or specific filtered data from the database is needed, use the `execute_sql_query` tool.
    If a general description, summary, or narrative insight is needed, use the `retrieve_relevant_chunks` tool.
    If the question falls under both categories above, use the `get_sql_rag_result` tool.
    If the question cannot be answered by these tools or is outside the scope of crash reports, state that.

    Output Format:
    - If using the `execute_sql_query` tool: `function_name` = execute_sql_query, `function_arguments` = ['YOUR_SQL_QUERY_HERE;']
    - If using the `retrieve_relevant_chunks` tool: `function_name` = retrieve_relevant_chunks, `function_arguments` = ['YOUR_NATURAL_LANGUAGE_QUERY_HERE']
    - If using both tools: `function_name` = get_sql_rag_result, `function_arguments` = ['YOUR_SQL_QUERY_HERE;', 'YOUR_NATURAL_LANGUAGE_QUERY_HERE']
    - If no tool is suitable: `function_name` = no_tool_needed, `function_arguments` = ['explain why']

    User Question: "{user_query}"
    """
    print(f"\n--- Gemini's Tool Decision Prompt ---\n{decision_prompt}\n----------------------------------")
    try:
        response = gemini_client.models.generate_content(
            model=GEN_MODEL,
            contents=decision_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Tool
            )
        )
        tool_instruction: Tool = response.parsed
        print(f"Gemini's decision: {tool_instruction.function_name}\n Args: {tool_instruction.function_arguments}")

        if tool_instruction.function_name == 'execute_sql_query':
            sql_query = tool_instruction.function_arguments[0].strip("'\"") # Remove potential quotes around the query
            return get_sql_tool_response(sql_query, user_query)

        elif tool_instruction.function_name == 'retrieve_relevant_chunks':
            rag_query = tool_instruction.function_arguments[0].strip("'\"") # Remove potential quotes
            return get_rag_tool_response(rag_query, user_query)
        
        elif tool_instruction.function_name == 'get_sql_rag_result':
            sql_query = tool_instruction.function_arguments[0].strip("'\"")
            rag_query = tool_instruction.function_arguments[1].strip("'\"")
            return get_sql_rag_tool_response(sql_query, rag_query, user_query)
        
        elif tool_instruction.function_name == 'no_tool_needed':
            return tool_instruction.function_arguments[0].strip("'\"")
        
        else:
            return "Error: Unknown tool function identified by Gemini."

    except Exception as e:
        return f"An error occurred while retriving the data: {e}"

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
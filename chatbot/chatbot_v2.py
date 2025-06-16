from google import genai
from google.genai import types
import sqlite3
import json
import chromadb

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
API_KEY = config['general']['api_key']
CHROMADB_PATH = config['paths']['chromadb_path']
CHROMA_COLLECTION = config['general']['chroma_collection']
DB_FILE = config['paths']['db_file']
DB_TABLE_NAME = config['general']['db_table_name'] 
EMBEDDING_MODEL = config['models']['004'] 
GEN_MODEL = config['models']['1.5-flash']

client = genai.Client(api_key=API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

def get_table_schema_dict(db_path: str, table_name: str, include_samples: bool = True, sample_limit: int = 1) -> dict:
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
    lines = [f"Table: {schema_dict['table_name']}", "Columns:"]
    for col in schema_dict["columns"]:
        sample = f" e.g., {', '.join(str(s) for s in col['sample'])}" if col.get("sample") else ""
        lines.append(f'- {col["name"]} ({col["type"]}){sample}')
    return "\n".join(lines)

schema = get_table_schema_dict(DB_FILE, DB_TABLE_NAME, True, 3)
formatted_schema = format_schema(schema)
DB_SCHEMA_DESCRIPTION = formatted_schema

def get_embedding(text: str):
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
Table Schema:
{DB_SCHEMA_DESCRIPTION}
Example SQL queries:
- Count crashes in 2020: `SELECT COUNT(*) FROM {DB_TABLE_NAME} WHERE CAST(crash_case_year AS INTEGER) = 2020`
- List all different kinds of incidents: `SELECT DISTINCT crash_type_description FROM {DB_TABLE_NAME}`
- What is the average count of serious injuries?: `SELECT AVG(number_of_severe_injuries) FROM {DB_TABLE_NAME}`

Important:
- Be precise with column names as listed in the schema.
- If the query cannot be answered by SQL, do not generate a SQL query.
"""

def retrieve_relevant_chunks(query: str, top_k: int = 5):
    """
    Retrieves relevant textual chunks from ChromaDB.
    """
    try:
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return {"status": "error", "message": "Could not generate embedding for query."}

        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents'] # Only retrieve the document text
        )
        relevant_docs = results['documents'][0]
        return {"status": "success", "data": relevant_docs}
    except Exception as e:
        return {"status": "error", "message": f"ChromaDB retrieval error: {e}"}
    
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

    Based on the user's question, determine which tool to use and how to call it.
    If a precise count, sum, average, or specific filtered data from the database is needed, use the `execute_sql_query` tool.
    If a general description, summary, or narrative insight is needed, use the `retrieve_relevant_chunks` tool.
    If the question cannot be answered by these tools or is outside the scope of crash reports, state that.

    Output Format:
    - If using SQL: `TOOL_CALL: execute_sql_query('YOUR_SQL_QUERY_HERE')`
    - If using RAG: `TOOL_CALL: retrieve_relevant_chunks('YOUR_NATURAL_LANGUAGE_QUERY_HERE')`
    - If no tool is suitable: `NO_TOOL_NEEDED` or `I cannot answer this question.`

    User Question: "{user_query}"
    """
    print(f"\n--- Gemini's Tool Decision Prompt ---\n{decision_prompt}\n----------------------------------")
    try:
        response = client.models.generate_content(
            model=GEN_MODEL,
            contents=decision_prompt,
        )
        tool_instruction = response.text.strip()
        print(f"Gemini's decision: {tool_instruction}")

        if tool_instruction.startswith("TOOL_CALL:"):
            # Parse the tool call (simple parsing, be careful with complex SQL)
            call_string = tool_instruction.replace("TOOL_CALL: ", "").strip()
            # Find the function name and arguments
            parts = call_string.split('(', 1) # Split only on the first '('
            func_name = parts[0].strip()
            args_str = parts[1].rstrip(')').strip()

            if func_name == 'execute_sql_query':
                # Extract the SQL query string
                sql_query = args_str.strip("'\"") # Remove potential quotes around the query
                print(f"Executing SQL Query: {sql_query}")
                tool_result = execute_sql_query(sql_query)
                print(f"SQL Tool Result: {tool_result}")

                # Now, ask Gemini to summarize the SQL result
                summary_prompt = f"""
                The user asked: "{user_query}"
                I executed a SQL query and got the following result:
                {json.dumps(tool_result['data'], indent=2)}

                Please provide a concise and clear answer to the user's question based on these results.
                If the query resulted in an error or empty data, state that no information was found.
                """
                final_answer_response = response = client.models.generate_content(
                                                        model=GEN_MODEL,
                                                        contents=summary_prompt,
                                                    )
                return final_answer_response.text

            elif func_name == 'retrieve_relevant_chunks':
                # Extract the RAG query string
                rag_query = args_str.strip("'\"") # Remove potential quotes
                print(f"Executing RAG Query for: {rag_query}")
                tool_result = retrieve_relevant_chunks(rag_query)
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
                    final_answer_response = response = client.models.generate_content(
                                                        model=GEN_MODEL,
                                                        contents=final_answer_prompt,
                                                    )
                    return final_answer_response.text
                else:
                    return f"I couldn't find relevant information for your question via RAG: {tool_result.get('message', '')}"

            else:
                return "Error: Unknown tool function identified by Gemini."

        elif tool_instruction == "NO_TOOL_NEEDED" or tool_instruction.startswith("I cannot answer"):
            return tool_instruction # Gemini decided not to use a tool or couldn't answer
        else:
            return "Error: Unexpected output from Gemini's tool decision. " + tool_instruction

    except Exception as e:
        return f"An error occurred during tool orchestration: {e}"

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
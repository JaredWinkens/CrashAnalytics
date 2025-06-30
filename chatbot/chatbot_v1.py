from google import genai
from google.genai import types
from google.genai.errors import ServerError
import sqlite3
import json
import concurrent.futures
import time

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
API_KEY = config['general']['api_key']
DB_FILE = config['paths']['db_file']
DB_TABLE_NAME = "Intersection" 
GEN_MODEL = config['models']['1.5-flash']
MAX_TOKENS = 250_000

client = genai.Client(api_key=API_KEY)

# -------------------------
# Define LLM System Instructions
# -------------------------
TRANSLATOR_MODULE_ROLE = """
You are an expert SQL generator.

Your primary task is to translate natural language questions into accurate, executable SQL queries.

You will be provided with the schema for each table in the following format:
Table: <table name>
Columns:
- <column name> (data type) e.g., <sample data>

**Guidelines:**
- **Strict Adherence to Schema:** You MUST strictly adhere to the provided schema. Do not invent tables or columns.
- **Data Formatting from Samples:** When filtering or comparing data, **always use the format demonstrated by the sample data**.
- **Accuracy and Executability:** Generate SQL that is syntactically correct and will execute successfully against the provided schema.
- **Clarity and Simplicity:** Prefer clear, straightforward SQL constructs over overly complex ones, as long as accuracy is maintained.
- **No Explanations or Prose:** Output ONLY the SQL query. Do not include any natural language explanations, apologies, or conversational filler.
- **Handle Ambiguity Gracefully:** If a user's question is truly ambiguous given the schema, try to make a reasonable assumption and generate the most probable query. If absolutely impossible to resolve, you may output the reason why starting with the character `#`.
- **Case Sensitivity:** Assume table and column names are case-sensitive as per the schema.
- **Data Types:** Be mindful of data types when applying filters or performing operations (e.g., use quotes for string comparisons, numerical comparisons for numbers).
- **No `SELECT *` for production queries:** Always specify column names explicitly in `SELECT` clauses, unless the user specifically asks for all columns or it's a simple `COUNT(*)` or similar aggregation.
"""

OUTPUT_MODULE_ROLE = """
You are an intelligent data interpretation assistant. 

Your primary role is to transform raw SQL query results into clear, concise, and informative natural language outputs or structured summaries, making the data easily understandable for the end-user.

**Guidelines:**
- **Focus on the User's Intent:** Always refer back to the original question to ensure your output directly answers it.
- **Clarity and Simplicity:** Avoid technical jargon. Explain the data in simple terms.
- **Summarize Key Findings:** Do not just list the data. Extract the most important information.
- **Handle Empty Results Gracefully:** If the query returns no data, inform the user appropriately.
- **Maintain Context:** If a previous turn's information is relevant, incorporate it.
- **Output Format:** Prioritize readability. Use bullet points, short paragraphs, or tables as appropriate for the data.
"""

def build_translator_prompt(user_question: str, schema_blocks: list, few_shot_examples: list) -> str:
    
    # Format few-shot examples
    few_shot_block = ""
    for i, ex in enumerate(few_shot_examples):
        few_shot_block += f"### Example {i+1}\n"
        few_shot_block += f"Natural language: {ex['text_input']}\n"
        few_shot_block += f"SQL: {ex['output']}\n\n"
    
    # Schema block
    full_schema = "\n\n".join(schema_blocks)

    # Final user prompt
    user_prompt = f"""{few_shot_block}### Now your turn
    Natural language: {user_question}
    SQL:"""
    
    return f"{full_schema}\n\n{user_prompt}"

def build_output_prompt(user_prompt: str, sql_query: str, query_result: str) -> str:

    original_user_pompt = f"""**Original User Question:**
    {user_prompt}"""

    query_executed = f"""**SQL Query Executed:** 
    {sql_query}"""

    result = f"""**Query Result:**
    {query_result}"""

    return f'{original_user_pompt}\n\n{query_executed}\n\n{result}'

def get_n_shot_examples(n: int) -> list:
    file = open("data/Chatbot_Training_Data.json")
    json_file = list(json.load(file))
    return json_file[:n]

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

def naturallang_to_sql(prompt) -> str:
    result = ""
    try:
        # Human-to-SQL Translator Module
        translator_response = client.models.generate_content(
            model=GEN_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=TRANSLATOR_MODULE_ROLE,
                temperature=0.3,
                top_k=5,
                top_p=0.7,
                ),
            contents=[prompt]
        )
        result = translator_response.text
    except ServerError as e:
        result = str(e)
        print(f"Server Error: {e}")
    except Exception as e:
        result = str(e)
        print(f"An unexpected error occurred: {e}")

    print("TRANSLATOR REPSONSE: ", result)
    return result

def clean_query(raw_query) -> str:
    query = raw_query.strip()
    query = query.strip("```sql")
    query = query.strip()
    return query

def trim_input_string(text: str) -> str:
    tokens = text.split()
    if len(tokens) > MAX_TOKENS:
        return str(tokens[:MAX_TOKENS])
    return text

def execute_sql_query(query, params=None):
    conn = None
    try:
        # Execute SQL Query
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        print(f"Executing query in thread: {query}")
        start_time = time.time()
        cursor.execute(query, params if params is not None else ())
        result = str(cursor.fetchall())
        end_time = time.time()
        print(f"Query executed in {end_time - start_time:.2f} seconds.")
        # result = call_with_timeout(pd.read_sql_query, 10, query, conn)
        # query_result = str(result)
        return result
    except Exception as e:
        raise e
    finally:
        if conn:
            conn.close()

def get_query_result(query) -> str:
    query_result = ""
    if (query.startswith('#')):
        query_result = query
    else:
        query_result = call_with_timeout(execute_sql_query, 60, query)
    
    trimmed_query_result = trim_input_string(query_result)
    print("QUERY RESULT: ", trimmed_query_result)
    return trimmed_query_result

def get_formatted_output(prompt):
    result = ""
    try:
        output_response = client.models.generate_content(
            model=GEN_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=OUTPUT_MODULE_ROLE,
                temperature=0.4,
                top_k=30,
                top_p=0.9,
            ),
            contents=[prompt]
        )
        result = output_response.text
    except ServerError as e:
        result = str(e)
        print(f"Server Error: {e}")
    except Exception as e:
        result = str(e)
        print(f"An unexpected error occurred: {e}")

    print("OUTPUT: ",result)
    return result

def generate_response(user_input):
    # Get name & data type of each column in the table
    schema = get_table_schema_dict(DB_FILE, DB_TABLE_NAME, True, 3)

    # Format schema
    schema_str = format_schema(schema)

    # Get n shot exmaples
    few_shot_examples = get_n_shot_examples(3)

    # Build prompt
    translator_prompt = build_translator_prompt(user_input, [schema_str], few_shot_examples)
    
    # Translate natural language to SQL
    translator_response = naturallang_to_sql(translator_prompt)

    # Clean up translator response
    query = clean_query(translator_response)

    # Get the result of the query
    query_result = get_query_result(query)

    # Build prompt
    output_prompt = build_output_prompt(user_input, query, query_result)
    
    # Get formatted output
    output = get_formatted_output(output_prompt)

    return output
    

def main():

    while (True):

        user_input = input("Type Your Question (\q to quit): ")

        if (user_input == "\q"):
            break

        generate_response(user_input)
    
if __name__ == "__main__":
    main()
    


    
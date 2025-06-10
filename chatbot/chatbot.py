from google import genai
from google.genai import types
from google.genai.errors import ServerError
import sqlite3
import json
from chatbot.chatbot_config import *
from chatbot.chatbot_preprocess import *
import concurrent.futures
import time

client = genai.Client(api_key=API_KEY)

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
            model=MODEL,
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
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS]
    return text

def execute_sql_query(query, params=None):
    conn = None
    try:
        # Execute SQL Query
        conn = sqlite3.connect(DATABASE)
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
            model=MODEL,
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
    schema = get_table_schema_dict(DATABASE, TABLE_NAME, True, 3)

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
    


    
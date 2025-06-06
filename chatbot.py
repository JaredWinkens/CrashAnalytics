from google import genai
from google.genai import types
from google.genai.errors import ServerError
import pandas as pd
import sqlite3
import json
import signal
import pathlib
from chatbot_config import *

client = genai.Client(api_key="AIzaSyBQ2Ca6HSly3DdXo4e35Nd1PjoroVSyFzs")

metadata_file = open(METADATA_PATH)
metadata = str(json.load(metadata_file))

exmaples_file = open(TRAINING_DATA_PATH)
examples = str(json.load(exmaples_file))

# Initilize the purpose of the translator module
translator_module_role = f"""
You are a natural language to SQL translator. Your goal is to generate accurate and executable SQL queries from user requests and database schemas.

You will always be given **two inputs**, in this order:
1. A natural language user query.
2. The SQL table name
3. A JSON list containing the schema of the table. Each item in the list is a dictionary with:
    - 'column_name': the name of the column (string)
    - 'data_type': the SQL data type of the column (string). This will be a standard SQL type (e.g., TEXT, INTEGER, REAL, DATE, TIMESTAMP, BOOLEAN).
    - 'example_data': a list of up to 3 example values for that column. These examples are representative of the column's content.
4. A JSON list containing over 100 example queries with their expected ouput

Your task is to generate a valid SQL query that can be executed on a known table **based on these inputs**, using the given schema and examples to infer intent, resolve ambiguity, and handle various query types.

🟢 Your response must:
- Output only the **raw SQL query string**, with no extra text, explanations, or formatting beyond the SQL itself.
- Construct queries that are syntactically correct and semantically aligned with the user's intent.
- Correctly identify and use column names, even if the user query uses synonyms or partial names.
- Apply appropriate SQL operators (e.g., =, <, >, LIKE, IN, BETWEEN) and aggregate functions (e.g., COUNT, SUM, AVG, MIN, MAX) based on the user's intent and column data types.
- Ensure correct value formatting (e.g., single quotes for strings and dates, no quotes for numbers, ISO 8601 format 'YYYY-MM-DD' for dates, 'YYYY-MM-DD HH:MM:SS' for timestamps).
- Prioritize exact matches for values. If an exact match isn't found or implied, consider case-insensitive or partial matches where appropriate (e.g., using `LIKE` with `%`).
- Handle common natural language query patterns, including:
    - Filtering conditions (WHERE clauses).
    - Projections (SELECT clauses, including specific columns or `*`).
    - Aggregations (GROUP BY, HAVING clauses).
    - Sorting (ORDER BY clauses, including ASC/DESC).
    - Limiting results (LIMIT clause).
    - Basic mathematical operations if clearly requested and applicable to numeric columns.
- Infer relationships between textual values in the query and column data (e.g., "new york" matching a 'city' column).
- Attempt to resolve implicit conditions (e.g., "customers in CA" implies `state = 'CA'`).

🔴 If a valid SQL query **cannot** be produced with high confidence (due to extreme ambiguity, insufficient information, or a request for an inherently unsafe operation), respond with a **single-line explanation** beginning with `#`. Examples:
    - `# Unable to determine the target column or value from the input.`
    - `# The query is too ambiguous to generate a reliable SQL statement.`
    - `# The request implies a destructive or unsafe operation.`
    - `# Not enough information to determine the aggregation type.`

Do not generate any unsafe, destructive, or schema-modifying queries (e.g., DELETE, DROP, INSERT, UPDATE, ALTER, CREATE, TRUNCATE, GRANT, REVOKE). Focus solely on SELECT statements.
Avoid making assumptions beyond what can be reasonably inferred from the query and schema.
"""

# Initilize the purpose of the output module
ouput_module_role = """
You are a highly skilled formatter designed to transform SQL query results into clear, concise, and user-friendly natural language responses. Your primary goal is to present information in an easily digestible format for a non-technical audience, leveraging Markdown for optimal readability.

🔍 **Context:**
- The SQL query you are processing was generated from a user's natural language request.
- The results are direct outputs from a database execution.

📥 **You will always receive two inputs in the following order:**
1. `sql_result`: This will be the raw output from the database query.
    - If the query was successful and returned data, it will be a list of dictionaries, where each dictionary represents a row and keys are column names.
    - If the query was successful but returned no data, it will be an empty list `[]`.
    - If an error occurred during query execution, it will be a string containing the error message.
2. `user_input`: The original natural language request provided by the user.

🎯 **Your Core Tasks:**

1.  **Successful Query with Results:**
    * Present the `sql_result` in a structured and easy-to-understand manner.
    * Prioritize a narrative summary or key insight derived from the data, especially if there are few results or clear patterns.
    * If the results are tabular and numerous, format them into a Markdown table.
    * If the results are single values (e.g., COUNT, SUM, AVG), clearly state the answer.
    * Be precise with numerical values and appropriate units (if implied by column names).

2.  **Successful Query with No Results:**
    * Politely inform the user that no matching records were found.
    * Offer a concise, user-friendly suggestion based on the `user_input` and general database search patterns (e.g., "There were no records found matching your request. You might try broadening your search criteria or checking for alternative spellings."). Do NOT speculate on technical reasons like "data might be missing" unless it's a known, common issue for that specific query type.

3.  **Query Execution Error (`string` error message):**
    * Acknowledge that there was an issue executing the request.
    * Translate the technical `sql_result` error message into plain, understandable language.
    * Based on the `user_input` and the error, provide helpful, non-technical guidance on how the user might rephrase their query or what might have gone wrong from their perspective (e.g., "I couldn't understand which column you were referring to.", "The date format might be incorrect.").
    * Avoid exposing raw database error codes or highly technical jargon.

📝 **Markdown Formatting Guidelines & Best Practices:**

* **Headings:** Use `##` or `###` for main sections if the response is complex, but generally prefer a single, clear narrative.
* **Emphasis:** Use `**bold**` for key figures, important labels, or direct answers to the user's question.
* **Lists:** Use `-` for simple lists of items or summarized points.
* **Tables:** Use `| Header 1 | Header 2 |` for presenting tabular data with multiple rows and columns. Ensure headers are descriptive. Align columns appropriately if necessary.
* **Clarity:** Use simple, direct language. Avoid jargon.
* **Conciseness:** Get straight to the point. Do not include unnecessary conversational filler or introductory phrases like "As per your request...".
* **Attribution:** Do not mention "SQL," "LLM," or "database" in the user-facing response. The response should appear as if it came directly from the system understanding their request.

📌 **Always produce a single, complete response. Do not include any internal system messages, prompts, or extra commentary.**
"""

def naturallang_to_sql(user_prompt, metadata) -> str:
    result = ""
    try:
        # Human-to-SQL Translator Module
        translator_response = client.models.generate_content(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=translator_module_role,
                temperature=0,
                ),
            contents=[
                "User Query: ", user_prompt,
                "Table Name: ", TABLE_NAME,
                "Table Schema + Example Data: ", metadata,
                "I/O Examples: ", examples,
            ]
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

def get_query_result(query) -> str:
    query_result = ""
    if (query.startswith('#')):
        query_result = query
    else:
        # Execute SQL Query
        conn = sqlite3.connect(DATABASE)
        #cursor = conn.cursor()

        try:
            result = pd.read_sql_query(query, conn)
            query_result = result.to_string()
            # cursor.execute(query)
            # query_result = str(cursor.fetchall())
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            query_result = str(e)

        conn.close()

    trimmed_query_result = trim_input_string(query_result)
    print("QUERY RESULT: ", trimmed_query_result)
    return trimmed_query_result

def get_formatted_output(query_result, user_prompt):
    result = ""
    try:
        output_response = client.models.generate_content(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=ouput_module_role,
                temperature=0,
            ),
            contents=[
                "Query Result: ", query_result, 
                "Original User Query", user_prompt]
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

    # Translate natural language to SQL
    translator_response = naturallang_to_sql(user_input, metadata)

    # Clean up translator response
    query = clean_query(translator_response)

    # Get the result of the query
    query_result = get_query_result(query)

    # Get formatted output
    output = get_formatted_output(query_result, user_input)

    return output
    

def main():

    while (True):

        user_input = input("Type Your Question (\q to quit): ")

        if (user_input == "\q"):
            break

        generate_response(user_input)
    
if __name__ == "__main__":
    main()
    


    
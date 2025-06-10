import pandas as pd
import sqlite3
from chatbot.chatbot_config import *
import json

def get_new_col_names():
    new_col_names = []
    file = open("data/Combined_Data_Col_Rename.json")
    json_file = json.load(file)
    for obj in json_file:
        new_col_names.append(obj['new_name'])
    print(new_col_names)
    return new_col_names

# Preprocessing functions
def csv_to_sql_table(db_path: str, table: str, csv_path: str, new_col_names: list = None) -> None:
    df = pd.read_csv(csv_path, header=0, parse_dates=['crash_date', 'dmv_insert_date'], names=new_col_names)
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists='replace', index=False)
    conn.close()
    print("Database Created Successfully")

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

def get_n_shot_examples(n: int) -> list:
    file = open("data/Chatbot_Training_Data.json")
    json_file = list(json.load(file))
    return json_file[:n]

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

def main():
    print("Preprocessing started")

    # Convert CSV to SQL table
    csv_to_sql_table(DATABASE, TABLE_NAME, CSV_FILE)

if __name__ == "__main__":
    main()
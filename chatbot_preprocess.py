import pandas as pd
import sqlite3
import json
from chatbot_config import *

df = pd.read_csv(CSV_FILE, parse_dates=['CrashDate', 'DMVInsertD'])

# Preprocessing functions
def csv_to_sql_table():

    conn = sqlite3.connect(DATABASE)
    
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)

    conn.close()
    print("Data imported successfully")

def get_sql_table_schema():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    table_info = cursor.fetchall()
    conn.close()

    schema = []
    for row in table_info:
        schema.append({
            'column_name': row[1],
            'data_type': row[2],
            'example_data': []
        })

    for i, col in enumerate(df.columns):
        first_three_values = df[col].head(3)
        #print(df[col].head(3))
        for val in first_three_values:
            curr_col = schema[i]
            curr_col['example_data'].append(val)
    
    return schema

def schema_to_json_file(schema):
    try:
        with open(METADATA_PATH, 'w') as jsonfile:
            json.dump(schema, jsonfile, indent=4)
        print(f"SQL schema saved to '{METADATA_PATH}'")
    except Exception as e:
        print(f"Error saving schema to JSON: {e}")
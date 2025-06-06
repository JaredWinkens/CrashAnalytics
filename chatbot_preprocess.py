import pandas as pd
import sqlite3
from chatbot_config import *

df = pd.read_csv(CSV_FILE, parse_dates=['CrashDate', 'DMVInsertD'])

# Preprocessing functions
def csv_to_sql_table():
    conn = sqlite3.connect(DATABASE)
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)
    conn.close()
    print("Database Created Successfully")

def get_sql_table_schema():
    schema = []

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
    table_info = cursor.fetchall()
    conn.close()

    for row in table_info:
        schema.append({
            'column_name': row[1],
            'data_type': row[2],
            'sample_data': []
        })

    for i, col in enumerate(df.columns):
        random_values = df[col].sample(n=3)
        for val in random_values:
            curr_col = schema[i]
            curr_col['sample_data'].append(val)
    
    return schema

def schema_to_json_file(schema):
    # Manually format the JSON string to get the desired output
    json_string = "[\n"
    for i, item in enumerate(schema):
        # Use json.dumps to format each dictionary with the desired indent
        #item_json = json.dumps(item, indent=4, skipkeys=True) # indent=4 for 4 spaces
        json_string += f"    {item}" # Add an extra 4 spaces for the array elements
        if i < len(schema) - 1:
            json_string += ","
        json_string += "\n"
    json_string += "]\n"

    # Write the manually formatted string to the file
    with open(METADATA_PATH, "w") as f:
        f.write(json_string)

    print(f"Data saved to {METADATA_PATH}.json")

def main():
    # Convert CSV to SQL table
    csv_to_sql_table()

    # Get name, data type, and example data of each column in the table
    schema = get_sql_table_schema()

    # Save schema to file
    schema_to_json_file(schema)

if __name__ == "__main__":
    main()
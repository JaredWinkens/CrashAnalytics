from google import genai
from google.genai import types
from pydantic import BaseModel
import time
import pandas as pd
import sqlite3
import csv
import enum
from pandasql import sqldf

MODEL = "gemini-2.0-flash"
DATABASE = "data/crash_data.db"
CSV_FILE = "data/Combined_Data.csv"

client = genai.Client(api_key="AIzaSyBQ2Ca6HSly3DdXo4e35Nd1PjoroVSyFzs")

class Column(BaseModel):
    col_name: str
    first_value: str

def preprocess_data(csv_file):

    df = pd.read_csv(csv_file) #, usecols=usecols, dtype=dtype, header=0)
    schema = []
    # Iterate through columns
    for col in df.columns:
        name = col
        first_value = df[col].iloc[0]
        schema.append(Column(col_name=name,first_value=str(first_value)))
        print(name)

    return df, schema


def df_to_sql_table(df, db_file, table_name):

    conn = sqlite3.connect(db_file)
    
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    conn.close()  # Close the connection
    print("Data imported successfully")



def main():
    table_name = "combined_data"
    df, schema= preprocess_data(CSV_FILE)
    df_to_sql_table(df, DATABASE, table_name)

    # Initilize the model's purpose
    translator_module_role = f"""
    Your purpose is to convert user input into SQL queries that can be executed on an SQL table to retrieve data.

    The table name is `{table_name}`. The name and first value of each column in the table is as follows: {str(schema)}

    Make sure to only output the raw SQL code. Do not add any other characters or whitespace. 
    """
    ouput_module_role = """
    Your purpose is to transform raw SQL query results into a highly informative, user-friendly, and actionable format. 
    
    Focus on clarity, conciseness, and immediate comprehensibility for a non-technical audience.
    """
    while (True):

        user_input = input("Type Your Question (\q to quit): ")

        if (user_input == "\q"):
            break
        
        # Human-to-SQL Translator Module
        htosql_response = client.models.generate_content(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=translator_module_role,
                temperature=0),
            contents=[user_input]
        )
        print("HTSQL REPSONSE: ", htosql_response.text)

        # Execute SQL Query
        query = htosql_response.text.strip("```sql")
        #result = sqldf(query,locals())
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute(query)
        result = cursor.fetchall()

        conn.commit() # Commit the changes
        conn.close()  # Close the connection
        print("QUERY RESULT: ", result)

        # Output Formater Module
        output_response = client.models.generate_content(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=ouput_module_role,
                temperature=0.5,
            ),
            contents=[user_input, query, str(result)]
        )
        print(output_response.text)

    
if __name__ == "__main__":
    main()
    


    
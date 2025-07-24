import json

data = {
        "paths": {
            "db_file":  "./data/chatbot/crash_data.db",
            "chromadb_path": "./data/chatbot/chroma_crash_data",
            "dataset_config_path": "./chatbot/datasets.json",
            "training_data_path": "./chatbot/training_data.json"
        },
        "general": {
            "api_key": ""
        },
        "models": {
            "1.5-flash": "gemini-1.5-flash",
            "2.0-flash": "gemini-2.0-flash",
            "2.5-flash": "gemini-2.5-flash",
            "004": "models/text-embedding-004",
            "001": "models/embedding-001",
            "exp-03-07": "gemini-embedding-exp-03-07",
            "roadsafetybotv01": "tunedModels/roadsafetybotv01-nu8d9hs98a6u56uzddesjyv"
        }
    }

data['general']['api_key'] = input("Please enter your Google API key: ")

try:
    with open("config.json", "w", encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    print("Data successfully written to output.json")
except IOError as e:
    print(f"Error writing to file: {e}")




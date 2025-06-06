# Define Constants
DATABASE =  "data/crash_data.db"
MODEL = "gemini-1.5-flash"
CSV_FILE =  "data/Combined_Data.csv"
TABLE_NAME = "combined_data"
METADATA_PATH = "data/Combined_Data_Metadata.json"
TRAINING_DATA_PATH = "data/Chatbot_Training_Data.json"
MAX_TOKENS = 200_000
APPROX_CHARS_PER_TOKEN = 4
MAX_CHARS = MAX_TOKENS * APPROX_CHARS_PER_TOKEN
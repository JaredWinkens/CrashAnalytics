import time
from google import genai
from google.genai import types
import json

# load config settings
config_file = open("config.json", "r")
config = json.load(config_file)
API_KEY = config['general']['api_key']
TRAINING_DATA_PATH = config['paths']['training_data_path']

client = genai.Client(api_key=API_KEY) # Get the key from the GOOGLE_API_KEY env variable

for model_info in client.models.list():
    print(model_info.name)

# create tuning model
file = open(TRAINING_DATA_PATH)
training_dataset =  json.load(file)
# print(type(training_dataset))
# for i in training_dataset:
#     print(i['text_input'], " ", i['output'])

training_dataset=types.TuningDataset(
        examples=[
            types.TuningExample(
                text_input=i['text_input'],
                output=i['output'],
            )
            for i in training_dataset
        ],
    )

tuning_job = client.tunings.tune(
    base_model='models/gemini-1.5-flash-001-tuning',
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count= 50,
        batch_size=4,
        learning_rate=0.001,
        tuned_model_display_name="road-safety-bot-v0.1"
    )
)

# Get job name
job_name = tuning_job.name
print(f"Tuning job started: {job_name}")

# Poll until job completes
while True:
    job = client.tunings.get(name=job_name)

    print(f"Job status: {job.state.name}")

    if job.state.name == "JOB_STATE_SUCCEEDED":
        break
    elif job.state.name in {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}:
        raise RuntimeError(f"Tuning job failed or was cancelled: {job.state.name}")

    time.sleep(10)

# Use the tuned model
response = client.models.generate_content(
    model=job.tuned_model.model,
    contents='How many fatal crashes happended in Jan 2020?',
)
print(response.text)

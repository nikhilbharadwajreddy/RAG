import json

def load_secrets(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Update this path to your JSON file location
secrets = load_secrets('/Users/bharadwajreddy/Desktop/AI-Projects/sec.json')

OPENAI_KEY = secrets["openai_key"]

print(OPENAI_KEY)
import json
import requests

with open('data/requests.json', 'r') as f:
    for line in f:
        request = json.loads(line)
        response = requests.post('http://0.0.0.0/predict', headers={'Content-Type': 'application/json'}, json=request)

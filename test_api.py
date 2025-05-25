import requests

# API URL (running locally)
url = "http://127.0.0.1:8000/generate"

# Data to send (JSON)
data = {
    "prompt": "Once upon a time",
    "max_length": 50
}

# Send POST request
response = requests.post(url, json=data)

# Print the response
print("Status Code:", response.status_code)
print("Response JSON:", response.json())

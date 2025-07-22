import requests

# The URL of your running Flask API
API_URL = "http://127.0.0.1:5000/predict"

# IMPORTANT: Change this to the name of an image file in the same folder
IMAGE_PATH = "r3.jpg" 

try:
    with open(IMAGE_PATH, 'rb') as image_file:
        file_payload = {'file': image_file}
        response = requests.post(API_URL, files=file_payload)
        
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())

except FileNotFoundError:
    print(f"Error: The file '{IMAGE_PATH}' was not found.")
except requests.exceptions.RequestException as e:
    print(f"Error: Could not connect to the API server. Is it running? Details: {e}")
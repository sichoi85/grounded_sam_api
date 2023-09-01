import requests

# Set the URL of your endpoint
url = "http://0.0.0.0:8080/uploadImage/"
import os
# Specify the file you want to upload
cwd = os.path.dirname(os.path.abspath(__file__))

files = {'file': ('emptyroomtouse.jpg', open(os.path.join(cwd,"inputs/emptyroomtouse.jpg"), 'rb'))}

try:
    # Send the POST request
    response = requests.post(url, files=files)
    
    # Check the response
    if response.status_code == 200:
        data = response.json()
        masks = data['message']['masks']
        # Do something with the response data
        print("Masks:", masks)
    else:
        print("HTTP Error:", response.status_code, response.text)

except requests.exceptions.RequestException as e:
    print("Request Error:", str(e))

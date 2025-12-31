import requests
import urllib3
urllib3.disable_warnings()
# -------------------------
# 1. User Inputs (readable dict)
# -------------------------
features = {
    "AGE": 19,
    "SEX": 0,      # male=1, female=0
    "BMI": 27.9,
    "CHILDREN": 0,
    "SMOKER": 1,   # yes=1, no=0
    "REGION": 1    # southeast=2, northwest=3, southwest=1, northeast=0
}

serving_url = "https://172.17.0.73:32443/deployment/15/dep89/"
auth_token  = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoic2hhcm1pc3RoYS1jaG91ZGh1cnkiLCJ0eXBlIjoiYXBpIiwiaWQiOiIvZGVwbG95bWVudC9kZXA4OS8ifQ.mBdIl8PZoJp5zpHCHQabSumosAJ8zuKkpI6SmuQJGfM"

# -------------------------
# 2. Prepare payload
# -------------------------
payload = {
    "data": [list(features.values())]
}
#headers={'Authorization': auth_token}
headers = {
    "Authorization": f"Bearer {auth_token}",
    "Content-Type": "application/json"
}

# -------------------------
# 3. Send request
# -------------------------
response = requests.post(serving_url, json=payload, headers=headers, verify=False)

# -------------------------
# 4. Print response
# -------------------------
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)

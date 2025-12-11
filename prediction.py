import requests
import urllib3
urllib3.disable_warnings()
# -------------------------
# 1. User Inputs (readable dict)
# -------------------------
features = {
    "age": 30,
    "sex": 1,      # male=1, female=0
    "bmi": 27.5,
    "children": 2,
    "smoker": 1,   # yes=1, no=0
    "region": 2    # southeast=2, northwest=3, southwest=1, northeast=0
}

serving_url = ""
auth_token  = ""

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

import requests
import json
resp = requests.get("http://127.0.0.1:8000/api/latest")
data = resp.json()
for i, point in enumerate(data["chart_data"][-5:]):
    print(f"Point {i}: Price={point['price']}, Probas={point['prediction']['probabilities']}")

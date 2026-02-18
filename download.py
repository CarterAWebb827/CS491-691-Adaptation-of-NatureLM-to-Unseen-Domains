import requests
import os

url = "https://xeno-canto.org/api/3/recordings"

# Load API key
with open("key", "r") as f:
    key = f.readline().strip()

params = {
    "query": 'sp:"Ochotona princeps"',
    "key": key,
    "per_page": 100
}

response = requests.get(url, params=params)
data = response.json()

# print(data)

os.makedirs("pika_audio", exist_ok=True)

for rec in data["recordings"]:
    file_url = "https:" + rec["file"]
    filename = f'pika_{rec["id"]}.mp3'

    audio = requests.get(file_url).content
    with open(f"pika_audio/{filename}", "wb") as f:
        f.write(audio)

    # Save metadata
    with open("pika_metadata.csv", "a") as m:
        m.write(
            f'{rec["id"]},{rec["date"]},{rec["lat"]},{rec["lon"]},{rec["q"]}\n'
        )
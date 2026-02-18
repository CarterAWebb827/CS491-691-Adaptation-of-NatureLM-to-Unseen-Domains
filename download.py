import requests
import os
import csv

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

dataset = "xeno-canto"
os.makedirs(f"data/Pika/pika_audio/{dataset}", exist_ok=True)

for rec in data["recordings"]:
    file_url = rec["file"]

    # Normalize URL
    if file_url.startswith("//"):
        file_url = "https:" + file_url

    filename = f'pika_{rec["id"]}.mp3'

    audio = requests.get(file_url).content
    with open(f"data/Pika/pika_audio/{dataset}/{filename}", "wb") as f:
        f.write(audio)

    file_path = "data/Pika/pika_metadata.csv"
    write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
    
    with open(file_path, "a", newline="") as m:
        writer = csv.writer(m)

        if write_header:
            writer.writerow(["audio_path", "filename", "dataset", "id", "length", "date", "location", "elev", "lat", "lon", "type"])

        writer.writerow([
            file_path,
            filename,
            dataset,
            rec["id"],
            rec["length"],
            rec["date"],
            rec["loc"],
            rec["alt"],
            rec["lat"],
            rec["lon"],
            rec["type"]
        ])

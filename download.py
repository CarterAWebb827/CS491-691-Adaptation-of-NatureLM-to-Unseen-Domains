import requests
import os
from pathlib import Path

species = "Ochotona princeps"
url = f"https://xeno-canto.org/api/3/recordings?query={species.replace(' ', '+')}"
parent_dir = Path(__file__).resolve().parent
print(parent_dir)
with open(os.path.join(parent_dir, "key"), "r") as f:
    key = f.readline().strip()

data = requests.get(query=url, key=key).json()
print(data)
os.makedirs("pika_audio", exist_ok=True)

for rec in data["recordings"]:
    file_url = "https:" + rec["file"]
    filename = f'pika_{rec["id"]}.mp3'

    audio = requests.get(file_url).content
    with open(f"pika_audio/{filename}", "wb") as f:
        f.write(audio)

    # Save metadata
    with open("pika_metadata.csv", "a") as m:
        m.write(f'{rec["id"]},{rec["date"]},{rec["lat"]},{rec["lng"]},{rec["q"]}\n')

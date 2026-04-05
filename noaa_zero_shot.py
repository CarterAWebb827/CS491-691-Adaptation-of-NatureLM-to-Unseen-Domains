import os
import sys
from pathlib import Path
from huggingface_hub import login
import argparse
import torch
import numpy as np
import pandas as pd

# Handle imports based on environment
current_dir = Path.cwd()
naturelm_dir = Path(os.path.join(current_dir, "NatureLMaudio"))

from NatureLMaudio.NatureLM.config import Config
from NatureLMaudio.NatureLM.infer import Pipeline

noaa_dir = Path("/content/drive/MyDrive/RightWhaleData")

from noaa_dataset import RightWhaleDataset

token = input("Paste huggingface token here: ")

login(token=token)

def is_right_whale(prediction):
    #drop to lowercase
    prediction = prediction.strip().lower()
    #common names and species names for right whales
    if "right whale" in prediction or "north atlantic right whale" in prediction or "north pacific right whale" in prediction or "southern right whale" in prediction or "eubalaena glacialis" in prediction or "eubalaena japonica" in prediction or "eubalaena australis" in prediction:
        return True
    else:
        return False

def is_whale_call(prediction):
    #drop to lowercase
    prediction = prediction.strip().lower()
    if "yes" in prediction or "true" in prediction:
        return True
    else:
        return False

def main():
    print("Loading the dataset...")

    cfg_path = "/content/drive/MyDrive/NatureLMaudio/configs/inference.yml"
    cfg = Config.from_sources(cfg_path)

    noaa_train_dataset = RightWhaleDataset(cfg, split="train", root_dir=noaa_dir)
    noaa_valid_dataset = RightWhaleDataset(cfg, split="valid", root_dir=noaa_dir)
    noaa_test_dataset = RightWhaleDataset(cfg, split="test", root_dir=noaa_dir)
    
    #dataset splits are 0.1 test, 0.18 valid, 0.72 train per noaa_dataset.py
    noaa_dataset = noaa_test_dataset #making sure it works using smallest available split
    noaa_df = noaa_dataset.df.reset_index(drop=True).copy()

    print(f"NOAA DataFrame shape: {noaa_df.shape}")
    print(f"Columns: {noaa_df.columns.tolist()}")

    print("Preparing clip audio...")
    clip_audio = []
    count = 0
    for _, row in noaa_df.iterrows():
        count += 1
        audio = noaa_dataset.load_audio(
            row["audio_path"],
            start_time=row["clip_start_seconds"],
            end_time=row["clip_end_seconds"],
        )
        clip_audio.append(audio.numpy())
        print (f"Audio count {count}")

    print("Running the pipeline...")
    results_path = "/content/drive/MyDrive/RightWhaleResults"
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, "zero_shot_results.txt")
    results = []

    if not os.path.exists(results_file):
        print("Loading the pipeline...")
        infer_pipe = Pipeline(cfg_path=cfg_path)

        results = infer_pipe(clip_audio, noaa_df["instruction"])

        with open(results_file, "w") as handle:
            handle.write("\n".join(results) + "\n")

        print(f"File saved to: {results_file}")
    else:
        with open(results_file) as handle:
            for line in handle:
                results.append(line.rstrip())

    print(f"Number of results: {len(results)}")
    print(f"Number of samples: {len(noaa_df)}")

    total_correct = 0
    confidence_correct = {}
    confidence_total = {}
    prediction_right_whales = []
    prediction_whale_calls = []
    
    for index, result in enumerate(results):
        output = noaa_df.iloc[index]["output"].strip().lower()
        confidence = noaa_df.iloc[index]["detection_confidence"]
        audio_path = noaa_df.iloc[index]["audio_path"]
        chunk_start = noaa_df.iloc[index]["chunk_start_time"]
        chunk_end = noaa_df.iloc[index]["chunk_end_time"]
        clip_start = noaa_df.iloc[index]["clip_start_seconds"]
        clip_end = noaa_df.iloc[index]["clip_end_seconds"]
        detection_start = noaa_df.iloc[index]["detection_start_time"]
        detection_end = noaa_df.iloc[index]["detection_end_time"]

        confidence_total[confidence] = confidence_total.get(confidence, 0) + 1

        if result != "":
            if ":" in result:
                prediction = result.split(":", 1)[1].strip().lower()
            else:
                prediction = result.strip().lower()
        else:
            prediction = ""

        predicted_whale_call = is_whale_call(prediction) #check for 'yes' or 'no'
        predicted_right_whale = is_right_whale(prediction) #check for species/common names

        whale_call_predicted = ("right whale" in output) and (predicted_whale_call)
        right_whale_predicted = ("right whale" in output) and (predicted_right_whale)

        if whale_call_predicted or right_whale_predicted:
            total_correct += 1
            confidence_correct[confidence] = confidence_correct.get(confidence, 0) + 1

        prediction_right_whales.append(right_whale_predicted)
        prediction_whale_calls.append(whale_call_predicted)

        if index < 5:
            print(f"\nExample {index}:")
            print(f"Audio Path: {audio_path}:")
            print(f"Ground Truth: {output}")
            print(f"Detection confidence: {confidence}")
            print(f"Prediction output: {prediction}")
            print(f"Prediction by species name: {predicted_right_whale}")
            print(f"Prediction by whale call: {predicted_whale_call}")
            print(f"")

    accuracy = (total_correct / len(noaa_df)) * 100
    print(f"\nZero-Shot Accuracy: {accuracy}")

    print("\nAccuracy by Detection_Confidence:")
    for confidence in sorted(confidence_total.keys()):
        correct = confidence_correct.get(confidence, 0)
        total = confidence_total[confidence]
        confidence_accuracy = (correct / total) * 100 if total > 0 else 0.0
        print(f"{confidence}: {confidence_accuracy:.2f}% ({correct}/{total})")

    detailed_df = pd.DataFrame(
        {
            "audio_path": noaa_df["audio_path"],
            "chunk_start_time": noaa_df["chunk_start_time"],
            "chunk_end_time": noaa_df["chunk_end_time"],
            "detection_start_time": noaa_df["detection_start_time"],
            "detection_end_time": noaa_df["detection_end_time"],
            "clip_start_seconds": noaa_df["clip_start_seconds"],
            "clip_end_seconds": noaa_df["clip_end_seconds"],
            "detection_confidence": noaa_df["detection_confidence"],
            "ground_truth": noaa_df["output"],
            "prediction_right_whales": right_whale_predicted,
            "prediction_whale_calls": whale_call_predicted,
            "raw_result": results,
        }
    )
    #save to somewhere that is easily accessible, results_path is just in my google drive
    detailed_df.to_csv(os.path.join(results_path, "detailed_results.csv"), index=False)


if __name__ == "__main__":
    main()

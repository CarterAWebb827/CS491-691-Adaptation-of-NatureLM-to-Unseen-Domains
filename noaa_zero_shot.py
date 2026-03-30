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

def majority_vote(predictions):
    """Simple majority vote function"""
    if not predictions:
        return "none"
    
    counts = {}
    for pred in predictions:
        counts[pred] = counts.get(pred, 0) + 1
    
    max_count = -1
    most_common = None
    for pred, count in counts.items():
        if count > max_count:
            max_count = count
            most_common = pred
    
    return most_common

def is_right_whale(prediction):
    prediction = prediction.strip().lower()
    return (
        "right whale" in prediction
        or "north atlantic right whale" in prediction
    )

def main():
    print("Loading the dataset...")

    cfg_path = "/content/drive/MyDrive/NatureLMaudio/configs/inference.yml"
    cfg = Config.from_sources(cfg_path)

    noaa_dataset = RightWhaleDataset(cfg, root_dir=noaa_dir)
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
    results_path = os.path.join(current_dir, "outputs/naturelm_zeroshot_noaa/")
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

    for index, result in enumerate(results):
        ground_truth = noaa_df.iloc[index]["output"].strip().lower()
        confidence = noaa_df.iloc[index]["detection_confidence"]

        confidence_total[confidence] = confidence_total.get(confidence, 0) + 1

        if result != "":
            prediction = result.split(":", 1)[1].strip().lower() if ":" in result else result.strip().lower()
        else:
            prediction = ""

        predicted_right_whale = is_right_whale(prediction)
        is_correct = predicted_right_whale and ground_truth == "right whale"

        if is_correct:
            total_correct += 1
            confidence_correct[confidence] = confidence_correct.get(confidence, 0) + 1

        if index < 5:
            print(f"\nExample {index}:")
            print(f"Ground truth: {ground_truth}")
            print(f"Detection confidence: {confidence}")
            print(f"Prediction: {prediction}")
            print(f"Predicted right whale: {predicted_right_whale}")
            print(f"Correct: {is_correct}")

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
            "clip_start_seconds": noaa_df["clip_start_seconds"],
            "clip_end_seconds": noaa_df["clip_end_seconds"],
            "detection_confidence": noaa_df["detection_confidence"],
            "ground_truth": noaa_df["output"],
            "raw_result": results,
        }
    )
    detailed_df.to_csv(os.path.join(results_path, "detailed_results.csv"), index=False)


if __name__ == "__main__":
    main()

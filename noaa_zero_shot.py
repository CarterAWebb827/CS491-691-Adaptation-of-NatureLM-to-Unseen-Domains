import os
from pathlib import Path
from huggingface_hub import login
import pandas as pd

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
    if (
        "right whale" in prediction or "north atlantic right whale" in prediction or "north pacific right whale" in prediction
        or "northern right whale" in prediction or "southern right whale" in prediction
        or "eubalaena glacialis" in prediction or "eubalaena japonica" in prediction or "eubalaena australis" in prediction
    ):
        return True
    else:
        return False

def compute_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn

    if total > 0:
        accuracy = ((tp + tn) / total) 
    else:
        accuracy = 0.0
    if (tp + fp) > 0:
        precision = (tp / (tp + fp)) 
    else:
        precision = 0.0
    if (tp + fn) > 0:
        recall = (tp / (tp + fn)) 
    else:
        recall = 0.0
    if (precision + recall > 0):
        f1_score = (2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}

def main():
    print("Loading the dataset...")

    cfg_path = "/content/drive/MyDrive/NatureLMaudio/configs/inference.yml"
    cfg = Config.from_sources(cfg_path)

    # Dataset construction prepares every split, so only the test split needs to be instantiated here.
    noaa_dataset = RightWhaleDataset(cfg, split="test", root_dir=noaa_dir)
    noaa_df = noaa_dataset.df.reset_index(drop=True).copy()

    print(f"NOAA DataFrame shape: {noaa_df.shape}")
    print(f"Columns: {noaa_df.columns.tolist()}")

    print("Preparing clip audio...")
    clip_audio = []
    count = 0
    for _, row in noaa_df.iterrows():
        count += 1
        audio = noaa_dataset.load_audio(row["audio_path"], start_time=row["clip_start_seconds"], end_time=row["clip_end_seconds"])
        clip_audio.append(audio.numpy())
        print (f"Audio count {count}")

    print("Running the pipeline...")
    results_path = "/content/drive/MyDrive/RightWhaleResults"
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, "zero_shot_results.txt")
    results = []

    if True:
        print("Loading the pipeline...")
        infer_pipe = Pipeline(cfg_path=cfg_path)

        results = infer_pipe(clip_audio, list(noaa_df["instruction"]))

        print(f"clip_audio: {len(clip_audio)}")
        print(f"instructions: {len(noaa_df['instruction'])}")
        print(f"results: {len(results)}")

        with open(results_file, "w") as handle:
            handle.write("\n".join(results) + "\n")
        print(f"File saved to: {results_file}")

    print(f"Number of results: {len(results)}")
    print(f"Number of samples: {len(noaa_df)}")

    if len(results) != len(noaa_df):
        raise ValueError(f"Expected {len(noaa_df)} results but received {len(results)}")

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    prediction_right_whales = []
    actual_right_whales = []
    confusion_labels = []

    for index, result in enumerate(results):
        row = noaa_df.iloc[index]
        output = row["output"].strip().lower()
        audio_path = row["audio_path"]
        chunk_start = row["chunk_start_time"]
        chunk_end = row["chunk_end_time"]
        actual_right_whale = bool(row[RightWhaleDataset.SPECIES_NAME])

        if result != "":
            if ":" in result:
                prediction = result.split(":", 1)[1].strip().lower() #take only the first result if multiple are found
            else:
                prediction = result.strip().lower()
        else:
            prediction = ""

        predicted_right_whale = is_right_whale(prediction) #check for species/common names

        if predicted_right_whale and actual_right_whale:
            tp += 1
            confusion_label = "TP"
        elif predicted_right_whale and not actual_right_whale:
            fp += 1
            confusion_label = "FP"
        elif (not predicted_right_whale) and actual_right_whale:
            fn += 1
            confusion_label = "FN"
        else:
            tn += 1
            confusion_label = "TN"

        prediction_right_whales.append(predicted_right_whale)
        actual_right_whales.append(actual_right_whale)
        confusion_labels.append(confusion_label)

        if index < 5:
            print(f"\nExample {index}:")
            print(f"Audio Path: {audio_path}:")
            print(f"Chunk window: {chunk_start} -> {chunk_end}")
            print(f"Ground Truth: {output}")
            print(f"Chunk label is right whale: {actual_right_whale}")
            print(f"Prediction output: {prediction}")
            print(f"Prediction by species name: {predicted_right_whale}")
            print(f"Confusion label: {confusion_label}")
            print(f"")

    metrics = compute_metrics(tp, fp, tn, fn)
    positive_chunks = int(noaa_df[RightWhaleDataset.SPECIES_NAME].sum())
    negative_chunks = len(noaa_df) - positive_chunks

    print(f"\nPositive chunks: {positive_chunks}")
    print(f"Negative chunks: {negative_chunks}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    metrics_row = {"total_samples": len(noaa_df), "positive_chunks": positive_chunks, "negative_chunks": negative_chunks, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    metrics_row["accuracy"] = metrics["accuracy"]
    metrics_row["precision"] = metrics["precision"]
    metrics_row["recall"] = metrics["recall"]
    metrics_row["f1_score"] = metrics["f1_score"]
    metrics_df = pd.DataFrame([metrics_row])
    metrics_df.to_csv(os.path.join(results_path, "metrics_summary.csv"), index=False)

    detailed_columns = {"audio_path": noaa_df["audio_path"], "chunk_start_time": noaa_df["chunk_start_time"], "chunk_end_time": noaa_df["chunk_end_time"]}
    detailed_columns["detection_start_time"] = noaa_df["detection_start_time"]
    detailed_columns["detection_end_time"] = noaa_df["detection_end_time"]
    detailed_columns["clip_start_seconds"] = noaa_df["clip_start_seconds"]
    detailed_columns["clip_end_seconds"] = noaa_df["clip_end_seconds"]
    detailed_columns["matching_detections"] = noaa_df["matching_detections"]
    detailed_columns["detection_confidence"] = noaa_df["detection_confidence"]
    detailed_columns["ground_truth"] = noaa_df["output"]
    detailed_columns["actual_right_whale"] = actual_right_whales
    detailed_columns["prediction_right_whales"] = prediction_right_whales
    detailed_columns["confusion_label"] = confusion_labels
    detailed_columns["raw_result"] = results
    detailed_df = pd.DataFrame(detailed_columns)
    #save to somewhere that is easily accessible, results_path is just in my google drive
    detailed_df.to_csv(os.path.join(results_path, "detailed_results.csv"), index=False)


if __name__ == "__main__":
    main()

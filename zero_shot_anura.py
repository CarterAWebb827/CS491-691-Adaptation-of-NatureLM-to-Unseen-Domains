import os
import gc
from pathlib import Path
from huggingface_hub import HfFolder, login
import argparse

token = HfFolder.get_token()
login(token=token)
del token
gc.collect()

current_dir = Path.cwd()
naturelm_dir = Path(os.path.join(current_dir, "NatureLMaudio"))

from NatureLMaudio.NatureLM.config import Config
from NatureLMaudio.NatureLM.infer import load_model_and_config
from NatureLMaudio.NatureLM.runner import Runner
from NatureLMaudio.NatureLM.infer import Pipeline

from xeno_dataset import XenoDataset

from anura_dataset import AnuraDataset

def get_anura_datasets(config, percentage):
    datasets = {}

    datasets["train"] = AnuraDataset(config=config, percentage=percentage, split="train")
    datasets["valid"] = AnuraDataset(config=config, percentage=percentage, split="valid")
    datasets["test"] = AnuraDataset(config=config, percentage=percentage, split="test")

    return datasets

def main():
    parser = argparse.ArgumentParser(description="A script to fine-tune the NatureLM-audio model on frog and toad species classification")
    parser.add_argument("--percentage", type=float, default=None, help="Designate the percentage of the full dataset used for fine-tuning")
    args = parser.parse_args()

    # Load our config
    cfg_path = "NatureLMaudio/configs/inference.yml"
    cfg = Config.from_sources(cfg_path)

    # Load in dataset
    print("Loading the dataset...")
    datasets = get_anura_datasets(cfg, args.percentage)

    # Run the pipeline
    print("Running the pipeline...")
    results_path = os.path.join(current_dir, "outputs/naturelm_zeroshot_anura/")
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, "results.txt")
    results = []

    if not os.path.exists(results_file):
        # Load the pipeline
        print("Loading the pipeline...")
        infer_pipe = Pipeline(cfg_path=cfg_path)

        # NOTE: We include instruction instead of instruction_text because it eventually gets passed to the generator,
        # which needs the location identifier (<Audio><AudioHere></Audio>) of where to place the audio embedding
        results = infer_pipe(datasets["test"].df["audio_path"], datasets["test"].df["instruction"])
        
        with open(results_file, "w") as f:
            f.write("\n".join(results) + "\n") # Write to the file, joining each result's time clips by new lines. Each result is then separated by a whole new line
        
        print(f"File saved to: {results_file}")
    else:
        with open(results_file) as f:
            for line in f:
                results.append(line.rstrip())

    # print(f"Number of results: {len(results)}")
    # print(f"Number of samples: {len(xeno_df)}")

    # Group the results by audio file
    grouped_results = []
    current_audio_windows = []
    
    for i, result in enumerate(results):
        # Parse the timestamp from the result
        if "#0.00s" in result and current_audio_windows: # Results are in form: #0.00s - 10.00s#: call
            # This means we are starting a new audio file
            if current_audio_windows:
                grouped_results.append(current_audio_windows)
            current_audio_windows = [result]
        else:
            # Continue with our current file
            current_audio_windows.append(result)
    
    if current_audio_windows:
        grouped_results.append(current_audio_windows) # Add the last result
    
    print(f"Grouped into {len(grouped_results)} audio files")

    def majority_vote(predictions):
        # Count occurences
        counts = {}
        for pred in predictions:
            counts[pred] = counts.get(pred, 0) + 1
        
        # Find the highest occuring prediction
        max_count = -1
        most_common = None
        for pred, count in counts.items():
            if count > max_count:
                max_count = count
                most_common = pred
        
        return most_common

    total_correct = 0
    for index, row in enumerate(grouped_results):
        ground_truth = datasets["test"].df.iloc[index]["output"].strip().lower()

        # Parse the window predictions
        window_preds = []
        for window_result in row:
            # Split at a ":", returning the next parsed string after it as 1 whole string,
            # removing white spaces and setting it to lower-case
            # print(f"{window_result}, {type(window_result)}")
            if window_result != "":
                prediction_list = window_result.split(":", 1)
                if len(prediction_list) > 1:
                    prediction_text = prediction_list[1].strip().lower()
                else:
                    prediction_text = prediction_list[0].strip().lower()
                predictions = []
                for pred in prediction_text.split(","):
                    predictions.append(pred.strip().lower())
                
                for pred in predictions:
                    # Make the labels lower-case
                    species_lower = []
                    for species in datasets["test"].label_columns:
                        species_lower.append(species.lower())
                    
                    if pred in species_lower or pred == "none":
                        window_preds.append(pred)
        
        # Apply simple majority vote
        if window_preds:  # Only vote if we have predictions
            most_common_preds = majority_vote(window_preds)
        else:
            most_common_preds = "none" # Default if no predictions

        if most_common_preds == ground_truth:
            total_correct += 1

        if index < 5: # Print first 5 examples
            print(f"\nExample {index}:")
            print(f"Ground truth: {ground_truth}")
            print(f"Window predictions: {window_preds}")
            print(f"Aggregated prediction (majority vote): {most_common_preds}")
            print(f"Correct: {most_common_preds == ground_truth}")
    
    accuracy = (total_correct / len(datasets["test"].df)) * 100
    print(f"\nZero-Shot Accuracy (majority vote): {accuracy}")

if __name__ == "__main__":
    main()
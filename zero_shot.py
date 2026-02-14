import os
import pandas as pd
from pathlib import Path

current_dir = Path.cwd()
beans_dir = Path(os.path.join(current_dir, "data/BEANS-Zero"))
naturelm_dir = Path(os.path.join(current_dir, "NatureLMaudio"))
# sys.path.append(naturelm_dir) # Appending to system path allows us to do "import infer" instead of "import NatureLMaudio.infer"

from NatureLMaudio.NatureLM.infer import Pipeline

from xeno_dataset import XenoDataset

def main():
    # Load our config
    cfg_path = "NatureLMaudio/configs/inference.yml"

    # Load in dataset
    print("Loading the dataset...")
    xeno_pth = os.path.join(beans_dir, "xeno_full.pkl")
    if not os.path.exists(xeno_pth):
        xeno_dataset = XenoDataset(beans_dir)
        xeno_df = xeno_dataset.get_full_df()
        xeno_df.to_pickle(xeno_pth)
    else:
        xeno_df = pd.read_pickle(xeno_pth)

    print(f"Full DataFrame shape: {xeno_df.shape}")
    print(f"Columns: {xeno_df.columns.tolist()}")

    # Run the pipeline
    print("Running the pipeline...")
    results_path = os.path.join(current_dir, "outputs/naturelm_zeroshot_calltype/")
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, "results.txt")
    results = []

    if not os.path.exists(results_file):
        # Load the pipeline
        print("Loading the pipeline...")
        infer_pipe = Pipeline(cfg_path=cfg_path)

        # NOTE: We include instruction instead of instruction_text because it eventually gets passed to the generator,
        # which needs the location identifier (<Audio><AudioHere></Audio>) of where to place the audio embedding
        results = infer_pipe(xeno_df["audio"], xeno_df["instruction"])
        
        with open(results_file, "w") as f:
            f.write("\n".join(results) + "\n") # Write to the file, joining each result's time clips by new lines. Each result is then separated by a whole new line
        
        print(f"File saved to: {results_file}")
    else:
        with open(results_file) as f:
            for line in f:
                results.append(line.rstrip())

    print(f"Number of results: {len(results)}")
    print(f"Number of samples: {len(xeno_df)}")

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
        ground_truth = xeno_df.iloc[index]["output"].strip().lower()

        # Parse the window predictions
        window_preds = []
        for window_result in row:
            # Split at a ":", returning the next parsed string after it as 1 whole string. It then returns the index 1 item of the returned string,
            # removing white spaces and setting it to lower-case
            # print(f"{window_result}, {type(window_result)}")
            if window_result != "":
                prediction = window_result.split(":", 1)[1].strip().lower() 
                # print(f"{prediction}, {type(prediction)}")
                if prediction in ["call", "song"]:
                    window_preds.append(prediction)
        
        # Apply simple majority vote
        most_common_pred = majority_vote(window_preds)

        if most_common_pred == ground_truth:
            total_correct += 1

        if index < 5: # Print first 5 examples
            print(f"\nExample {index}:")
            print(f"Ground truth: {ground_truth}")
            print(f"Window predictions: {window_preds}")
            print(f"Aggregated prediction (majority vote): {most_common_pred}")
            print(f"Correct: {most_common_pred == ground_truth}")
    
    accuracy = total_correct / len(xeno_df)
    print(f"\nZero-Shot Accuracy (majority vote): {accuracy}")

if __name__ == "__main__":
    main()
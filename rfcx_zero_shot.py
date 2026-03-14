import os
import gc
from pathlib import Path
from huggingface_hub import login
import argparse
import torch
import pandas as pd
import numpy as np

# Handle imports based on environment
current_dir = Path.cwd()
naturelm_dir = Path(os.path.join(current_dir, "NatureLMaudio"))

from NatureLMaudio.NatureLM.config import Config
from NatureLMaudio.NatureLM.infer import Pipeline

# Import our RFCx dataset
from rfcx_dataset import RFCFrogDataset, RFCTestDataset

login()

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

def evaluate_zero_shot_rfcx(config, data_root="data/rfcx", results_dir="outputs/naturelm_zeroshot_rfcx",
                           cache_results=True, num_examples_to_print=5):
    """
    Evaluate zero-shot performance on RFCx frog dataset
    
    Args:
        config: Configuration object for the model
        data_root: Root directory containing RFCx data
        results_dir: Directory to save results
        cache_results: Whether to cache results to disk
        num_examples_to_print: Number of example predictions to print
    
    Returns:
        dict: Dictionary containing accuracy and detailed results
    """
    
    # Create results directory
    results_path = Path(current_dir) / results_dir
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / "zero_shot_results.txt"
    
    # Load the test dataset (using train_tp as validation since test has no labels)
    # For zero-shot evaluation, we'll use a subset of TP data as validation
    print("Loading RFCx validation dataset (from train_tp)...")
    
    # Load TP data for evaluation
    tp_path = Path(data_root) / "train_tp.csv"
    tp_df = pd.read_csv(tp_path)
    
    # Filter for frog species (0-10)
    frog_species = list(range(11))
    tp_df = tp_df[tp_df['species_id'].isin(frog_species)].copy()
    
    # Take a random sample for zero-shot evaluation (e.g., 400 samples)
    eval_df = tp_df.sample(n=min(500, len(tp_df)), random_state=42)
    
    # Add required columns
    eval_df['audio_path'] = eval_df['recording_id'].apply(
        lambda x: str(Path(data_root) / "train" / f"{x}.flac")
    )
    eval_df['instruction'] = "<Audio><AudioHere></Audio> What is the scientific name for the frog species in the audio, if any?"
    eval_df['output'] = eval_df['species_id'].apply(
        lambda x: RFCFrogDataset.SPECIES_MAPPING[x][0]
    )
    
    print(f"Loaded {len(eval_df)} samples for zero-shot evaluation")
    
    # Check if we have cached results
    results = []
    if cache_results and results_file.exists():
        print(f"Loading cached results from {results_file}")
        with open(results_file, 'r') as f:
            for line in f:
                results.append(line.rstrip())
    else:
        # Load the pipeline
        print("Loading inference pipeline...")
        cfg_path = "NatureLMaudio/configs/inference.yml"
        infer_pipe = Pipeline(cfg_path=cfg_path)
        
        # Run inference on eval set
        print(f"Running zero-shot inference on {len(eval_df)} samples...")
        results = infer_pipe(eval_df["audio_path"], eval_df["instruction"])
        
        # Cache results if requested
        if cache_results:
            with open(results_file, 'w') as f:
                f.write("\n".join(results) + "\n")
            print(f"Results cached to: {results_file}")
    
    # Group results by audio file
    grouped_results = []
    current_audio_windows = []
    
    for i, result in enumerate(results):
        if "#0.00s" in result and current_audio_windows:
            if current_audio_windows:
                grouped_results.append(current_audio_windows)
            current_audio_windows = [result]
        else:
            current_audio_windows.append(result)
    
    if current_audio_windows:
        grouped_results.append(current_audio_windows)
    
    # Ensure we have the same number of groups as eval samples
    assert len(grouped_results) == len(eval_df), \
        f"Mismatch: {len(grouped_results)} groups vs {len(eval_df)} samples"
    
    print(f"Grouped into {len(grouped_results)} audio files")
    
    # Evaluate predictions
    detailed_results = []
    total_correct = 0
    
    # Get species mapping for reference
    species_mapping = RFCFrogDataset.SPECIES_MAPPING
    id_to_name = {k: v[0].lower() for k, v in species_mapping.items()}
    
    for idx, (row, window_results) in enumerate(zip(eval_df.iterrows(), grouped_results)):
        row_data = row[1]
        ground_truth = row_data["output"].strip().lower()
        species_id = row_data["species_id"]
        audio_path = row_data["audio_path"]
        
        # Parse window predictions
        window_preds = []
        for window_result in window_results:
            if window_result and window_result.strip():
                prediction_list = window_result.split(":", 1)
                if len(prediction_list) > 1:
                    prediction_text = prediction_list[1].strip().lower()
                else:
                    prediction_text = prediction_list[0].strip().lower()
                
                predictions = [p.strip().lower() for p in prediction_text.split(",")]
                
                for pred in predictions:
                    # Check if prediction matches any frog species
                    if pred in id_to_name.values() or pred == "none":
                        window_preds.append(pred)
        
        # Apply majority vote
        most_common_pred = majority_vote(window_preds)
        
        # Check correctness
        is_correct = (most_common_pred == ground_truth)
        if is_correct:
            total_correct += 1
        
        detailed_results.append({
            'index': idx,
            'recording_id': row_data['recording_id'],
            'species_id': species_id,
            'ground_truth': ground_truth,
            'window_predictions': window_preds,
            'aggregated_prediction': most_common_pred,
            'correct': is_correct
        })
        
        if idx < num_examples_to_print:
            print(f"\n{'='*50}")
            print(f"Example {idx}:")
            print(f"Recording ID: {row_data['recording_id']}")
            print(f"Species ID: {species_id} ({species_mapping[species_id][1]})")
            print(f"Ground truth: {ground_truth}")
            print(f"Window predictions ({len(window_preds)} windows): {window_preds}")
            print(f"Aggregated prediction: {most_common_pred}")
            print(f"Correct: {is_correct}")
    
    # Calculate accuracy
    accuracy = (total_correct / len(eval_df)) * 100
    
    # Calculate per-species accuracy
    species_accuracy = {}
    for species_id in frog_species:
        species_mask = eval_df['species_id'] == species_id
        if species_mask.any():
            species_indices = species_mask[species_mask].index.tolist()
            species_correct = sum(1 for r in detailed_results if r['index'] in species_indices and r['correct'])
            species_total = len(species_indices)
            species_name, species_code = species_mapping[species_id]
            species_accuracy[species_id] = {
                'species_name': species_name,
                'species_code': species_code,
                'accuracy': (species_correct / species_total) * 100,
                'correct': species_correct,
                'total': species_total
            }
    
    # Print summary
    print(f"\n{'='*50}")
    print("ZERO-SHOT EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {len(eval_df)}")
    print(f"Correct predictions: {total_correct}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    if species_accuracy:
        print(f"\nPer-species Accuracy:")
        for species_id, stats in species_accuracy.items():
            print(f"  {stats['species_code']} ({stats['species_name']}): "
                  f"{stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    
    print(f"{'='*50}")
    
    return {
        'accuracy': accuracy,
        'total_samples': len(eval_df),
        'correct_predictions': total_correct,
        'detailed_results': detailed_results,
        'species_accuracy': species_accuracy,
        'eval_df': eval_df
    }

def create_test_submission(config, model_path=None, data_root="data/rfcx", 
                          output_file="submission.csv", batch_size=8):
    """
    Create submission file for test set
    
    Args:
        config: Configuration object
        model_path: Path to fine-tuned model (None for zero-shot)
        data_root: Root directory containing RFCx data
        output_file: Output CSV file name
        batch_size: Batch size for inference
    """
    
    from NatureLMaudio.NatureLM.infer import Pipeline
    
    # Load test dataset
    test_dataset = RFCTestDataset(config, root_dir=data_root)
    
    # Load pipeline
    cfg_path = "NatureLMaudio/configs/inference.yml"
    if model_path:
        # Load fine-tuned model
        checkpoint = torch.load(model_path, map_location='cpu')
        model, _ = load_model_and_config(cfg_path=cfg_path, device=config.model.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        infer_pipe = Pipeline(model=model, cfg_path=cfg_path)
    else:
        # Zero-shot
        infer_pipe = Pipeline(cfg_path=cfg_path)
    
    # Load sample submission to get column structure
    submission_template = pd.read_csv(Path(data_root) / "sample_submission.csv")
    
    # Initialize submission dataframe with zeros
    submission_df = submission_template.copy()
    
    # Set all columns to 0 initially
    species_cols = [f's{i}' for i in range(24)]
    submission_df[species_cols] = 0
    
    # Group by recording_id and process in batches
    unique_recordings = submission_df['recording_id'].unique()
    
    print(f"Processing {len(unique_recordings)} test recordings...")
    
    for recording_id in unique_recordings:
        audio_path = Path(data_root) / "test" / f"{recording_id}.flac"
        
        # Get model prediction
        instruction = "<Audio><AudioHere></Audio> What are the scientific name(s) for the species in the audio, if any?"
        results = infer_pipe([str(audio_path)], [instruction])
        
        # Parse prediction
        prediction_text = results[0].lower() if results else "none"
        
        # Map prediction to species probabilities
        # For simplicity, we'll set probability 1.0 for predicted species, 0 for others
        # In practice, you might want to calibrate probabilities
        for species_id in range(11):
            species_name = RFCFrogDataset.SPECIES_MAPPING[species_id][0].lower()
            if species_name in prediction_text:
                # Species detected
                submission_df.loc[submission_df['recording_id'] == recording_id, f's{species_id}'] = 1.0
        
        # For species 11-23, always 0 (we're only classifying frogs)
        # They remain 0 from initialization
    
    # Save submission
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    
    return submission_df

def main():
    """Command-line interface for zero-shot evaluation"""
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on RFCx frog dataset")
    parser.add_argument("--data_root", type=str, default="data/rfcx", 
                       help="Root directory containing RFCx data")
    parser.add_argument("--results_dir", type=str, default="outputs/naturelm_zeroshot_rfcx",
                       help="Directory to save results")
    parser.add_argument("--no_cache", action="store_true",
                       help="Disable result caching")
    parser.add_argument("--num_examples", type=int, default=5,
                       help="Number of example predictions to print")
    parser.add_argument("--create_submission", action="store_true",
                       help="Create submission file for test set")
    parser.add_argument("--output_file", type=str, default="submission.csv",
                       help="Output file for submission")
    args = parser.parse_args()
    
    # Load configuration
    cfg_path = "NatureLMaudio/configs/inference.yml"
    cfg = Config.from_sources(cfg_path)
    
    if args.create_submission:
        # Create submission file
        create_test_submission(
            config=cfg,
            data_root=args.data_root,
            output_file=args.output_file
        )
    else:
        # Run zero-shot evaluation
        results = evaluate_zero_shot_rfcx(
            config=cfg,
            data_root=args.data_root,
            results_dir=args.results_dir,
            cache_results=not args.no_cache,
            num_examples_to_print=args.num_examples
        )
    
    return results if not args.create_submission else None

if __name__ == "__main__":
    main()
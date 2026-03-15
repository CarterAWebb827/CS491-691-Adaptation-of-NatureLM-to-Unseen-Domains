import os
from pathlib import Path
from huggingface_hub import login
import argparse
import pandas as pd
from datetime import datetime

# Handle imports based on environment
current_dir = Path.cwd()
naturelm_dir = Path(os.path.join(current_dir, "NatureLMaudio"))

from NatureLMaudio.NatureLM.config import Config
from NatureLMaudio.NatureLM.infer import Pipeline

# Import our Anura dataset
from anura_dataset import AnuraDataset

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

def evaluate_zero_shot(dataset, config, results_dir, dataset_name="test", cache_results=True, num_examples_to_print=5):
    """
    Evaluate zero-shot performance on a dataset
    
    Args:
        dataset: AnuraDataset object
        config: Configuration object for the model
        results_dir: Directory to save results
        dataset_name: Name of dataset being evaluated ("test" or "valid")
        cache_results: Whether to cache results to disk
        num_examples_to_print: Number of example predictions to print
    
    Returns:
        dict: Dictionary containing accuracy and detailed results
    """
    
    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_path / f"zero_shot_{dataset_name}_results_{timestamp}.txt"
    summary_file = results_path / f"zero_shot_{dataset_name}_summary_{timestamp}.txt"
    
    print(f"\nEvaluating zero-shot on {dataset_name} set: {len(dataset)} samples")
    
    # Create evaluation dataframe
    eval_data = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        row = dataset.df.iloc[idx]
        eval_data.append({
            'index': idx,
            'species_list': row[dataset.label_columns].values.tolist(),
            'output': item['text'],
            'audio_path': item['id'],
            'label_columns': dataset.label_columns,
            'split': row.get('split', 'unknown')
        })
    
    eval_df = pd.DataFrame(eval_data)
    
    # Check if we have cached results (optional, can use a fixed name for caching)
    cache_file = results_path / f"zero_shot_{dataset_name}_cached_results.txt"
    results = []
    
    if cache_results and cache_file.exists():
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, 'r') as f:
            for line in f:
                results.append(line.rstrip())
    else:
        # Load the pipeline
        print("Loading inference pipeline...")
        cfg_path = "NatureLMaudio/configs/inference.yml"
        infer_pipe = Pipeline(cfg_path=cfg_path)
        
        # Run inference in batches
        print(f"Running zero-shot inference on {len(eval_df)} samples...")
        batch_size = 8
        all_results = []
        
        for i in range(0, len(eval_df), batch_size):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(eval_df)}")
            
            batch_end = min(i + batch_size, len(eval_df))
            batch_paths = eval_df["audio_path"].iloc[i:batch_end].tolist()
            batch_instructions = ["<Audio><AudioHere></Audio> What are the scientific name(s) for the species in the audio, if any?"] * len(batch_paths)
            
            batch_results = infer_pipe(batch_paths, batch_instructions)
            all_results.extend(batch_results)
        
        results = all_results
        
        # Cache results if requested
        if cache_results:
            with open(cache_file, 'w') as f:
                f.write("\n".join(results) + "\n")
            print(f"Results cached to: {cache_file}")
    
    # Save raw results with timestamp
    with open(results_file, 'w') as f:
        f.write("\n".join(results) + "\n")
    print(f"Raw results saved to: {results_file}")
    
    # Group results by audio file (handling multiple windows)
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
    if len(grouped_results) != len(eval_df):
        print(f"Warning: Grouped results count ({len(grouped_results)}) doesn't match eval samples ({len(eval_df)})")
        # Pad with empty lists if necessary
        while len(grouped_results) < len(eval_df):
            grouped_results.append([])
    
    print(f"Grouped into {len(grouped_results)} audio files")
    
    # Evaluate predictions
    detailed_results = []
    total_correct_exact = 0
    total_correct_any = 0
    total_species_present = 0
    total_species_predicted = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Per-species metrics
    species_metrics = {col: {'tp': 0, 'fp': 0, 'fn': 0} for col in dataset.label_columns}
    
    for idx, (row, window_results) in enumerate(zip(eval_df.iterrows(), grouped_results)):
        row_data = row[1]
        ground_truth_text = row_data["output"].strip().lower()
        
        # Parse ground truth species
        if ground_truth_text == "none":
            ground_truth_species = set()
        else:
            ground_truth_species = set([s.strip() for s in ground_truth_text.split(",")])
        
        # Parse window predictions
        window_preds = []
        for window_result in window_results:
            if window_result and window_result.strip():
                prediction_list = window_result.split(":", 1)
                if len(prediction_list) > 1:
                    prediction_text = prediction_list[1].strip().lower()
                else:
                    prediction_text = prediction_list[0].strip().lower()
                
                # Handle multiple predictions
                if "," in prediction_text:
                    predictions = [p.strip().lower() for p in prediction_text.split(",")]
                else:
                    predictions = [prediction_text]
                
                for pred in predictions:
                    if pred != "none" and pred not in window_preds:
                        window_preds.append(pred)
        
        # For multi-label, we'll consider any prediction
        predicted_species = set(window_preds) if window_preds else set(["none"])
        
        # Calculate metrics
        exact_match = (predicted_species == ground_truth_species or 
                      (predicted_species == set(["none"]) and ground_truth_species == set()))
        if exact_match:
            total_correct_exact += 1
        
        # Any correct detection (if any predicted species is in ground truth)
        if ground_truth_species:
            any_correct = len(predicted_species.intersection(ground_truth_species)) > 0
            if any_correct:
                total_correct_any += 1
        
        # Per-species metrics
        for species in ground_truth_species:
            if species in predicted_species:
                species_metrics[species]['tp'] += 1
            else:
                species_metrics[species]['fn'] += 1
        
        for species in predicted_species - set(["none"]):
            if species not in ground_truth_species:
                species_metrics[species]['fp'] += 1
        
        tp = len(predicted_species.intersection(ground_truth_species))
        fp = len(predicted_species - ground_truth_species - set(["none"]))
        fn = len(ground_truth_species - predicted_species)
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        total_species_present += len(ground_truth_species)
        total_species_predicted += len(predicted_species - set(["none"]))
        
        detailed_results.append({
            'index': idx,
            'ground_truth': ground_truth_text,
            'ground_truth_set': list(ground_truth_species),
            'window_predictions': window_preds,
            'predicted_set': list(predicted_species - set(["none"])),
            'exact_match': exact_match,
            'any_correct': any_correct if ground_truth_species else None,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        })
        
        if idx < num_examples_to_print:
            print(f"\n{'='*50}")
            print(f"Example {idx}:")
            print(f"Ground truth: {ground_truth_text}")
            print(f"Window predictions ({len(window_preds)} windows): {window_preds}")
            print(f"Aggregated prediction: {', '.join(predicted_species - set(['none'])) if predicted_species - set(['none']) else 'none'}")
            print(f"Exact match: {exact_match}")
    
    # Calculate overall metrics
    exact_accuracy = (total_correct_exact / len(eval_df)) * 100
    any_accuracy = (total_correct_any / len(eval_df)) * 100 if total_species_present > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-species metrics
    per_species_results = {}
    for species, metrics in species_metrics.items():
        sp_precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
        sp_recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
        sp_f1 = 2 * (sp_precision * sp_recall) / (sp_precision + sp_recall) if (sp_precision + sp_recall) > 0 else 0
        
        per_species_results[species] = {
            'true_positives': metrics['tp'],
            'false_positives': metrics['fp'],
            'false_negatives': metrics['fn'],
            'precision': sp_precision,
            'recall': sp_recall,
            'f1': sp_f1
        }
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"ZERO-SHOT EVALUATION SUMMARY - {dataset_name.upper()} SET")
    print(f"{'='*50}")
    print(f"Total samples: {len(eval_df)}")
    print(f"Exact match accuracy: {exact_accuracy:.2f}%")
    print(f"Any correct detection accuracy: {any_accuracy:.2f}%")
    print(f"\nOverall metrics:")
    print(f"  True Positives: {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    
    # Save detailed summary to file
    with open(summary_file, "w") as f:
        f.write(f"ZERO-SHOT EVALUATION SUMMARY - {dataset_name.upper()} SET\n")
        f.write("="*50 + "\n")
        f.write(f"Total samples: {len(eval_df)}\n")
        f.write(f"Exact Match Accuracy: {exact_accuracy:.2f}%\n")
        f.write(f"Any Correct Detection Accuracy: {any_accuracy:.2f}%\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  True Positives: {true_positives}\n")
        f.write(f"  False Positives: {false_positives}\n")
        f.write(f"  False Negatives: {false_negatives}\n")
        f.write(f"  Precision: {precision:.3f}\n")
        f.write(f"  Recall: {recall:.3f}\n")
        f.write(f"  F1 Score: {f1:.3f}\n\n")
        
        f.write("Per-species Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Species':<30} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
        f.write("-" * 80 + "\n")
        
        for species, metrics in sorted(per_species_results.items()):
            f.write(f"{species:<30} {metrics['true_positives']:<6} {metrics['false_positives']:<6} "
                   f"{metrics['false_negatives']:<6} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                   f"{metrics['f1']:<10.3f}\n")
    
    print(f"\nEvaluation summary saved to: {summary_file}")
    
    # Save detailed predictions
    predictions_df = pd.DataFrame(detailed_results)
    predictions_file = results_path / f"zero_shot_{dataset_name}_predictions_{timestamp}.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Detailed predictions saved to: {predictions_file}")
    
    return {
        'exact_accuracy': exact_accuracy,
        'any_accuracy': any_accuracy,
        'total_samples': len(eval_df),
        'correct_predictions_exact': total_correct_exact,
        'correct_predictions_any': total_correct_any,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_species_metrics': per_species_results,
        'detailed_results': detailed_results
    }

def main():
    """Command-line interface for zero-shot evaluation"""
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on Anura frog dataset")
    parser.add_argument("--data_root", type=str, default="data/AnuraSet", 
                       help="Root directory containing Anura data")
    parser.add_argument("--results_dir", type=str, default="outputs/naturelm_zeroshot_anura",
                       help="Directory to save results")
    parser.add_argument("--no_cache", action="store_true",
                       help="Disable result caching")
    parser.add_argument("--num_examples", type=int, default=5,
                       help="Number of example predictions to print")
    parser.add_argument("--use_percentage", type=float, default=None,
                       help="Percentage of data to use (for quick testing)")
    parser.add_argument("--evaluate_test", action="store_true",
                       help="Evaluate on test set (default: evaluate on validation set)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for predictions (optional)")
    args = parser.parse_args()
    
    # Load configuration
    cfg_path = "NatureLMaudio/configs/inference.yml"
    cfg = Config.from_sources(cfg_path)
    
    # Determine which split to evaluate
    if args.evaluate_test:
        split_name = "test"
        print("\n" + "="*50)
        print("EVALUATING ZERO-SHOT ON TEST SET")
        print("="*50)
    else:
        split_name = "valid"
        print("\n" + "="*50)
        print("EVALUATING ZERO-SHOT ON VALIDATION SET")
        print("="*50)
    
    # Load the dataset
    print(f"Loading Anura {split_name} dataset...")
    dataset = AnuraDataset(
        config=cfg,
        split=split_name,
        root_dir=args.data_root,
        percentage=args.use_percentage
    )
    
    # Run zero-shot evaluation
    results = evaluate_zero_shot(
        dataset=dataset,
        config=cfg,
        results_dir=args.results_dir,
        dataset_name=split_name,
        cache_results=not args.no_cache,
        num_examples_to_print=args.num_examples
    )
    
    return results

if __name__ == "__main__":
    main()
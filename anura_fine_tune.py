import os
import sys
import gc
from pathlib import Path
from huggingface_hub import login
import argparse
import pandas as pd
import torch
import torch.cuda as cuda

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    # Add the NatureLMaudio directory to Python path
    current_dir = Path.cwd()
    naturelm_dir = current_dir / "NatureLMaudio"
    if str(naturelm_dir) not in sys.path:
        sys.path.insert(0, str(naturelm_dir))
        print(f"Added {naturelm_dir} to Python path")
    
    from NatureLM.config import Config
    from NatureLM.infer import load_model_and_config, Pipeline
    from NatureLM.runner import Runner
else:
    from NatureLMaudio.NatureLM.config import Config
    from NatureLMaudio.NatureLM.infer import load_model_and_config, Pipeline
    from NatureLMaudio.NatureLM.runner import Runner

login()

current_dir = Path.cwd()
naturelm_dir = Path(os.path.join(current_dir, "NatureLMaudio"))

# Import our Anura dataset
from anura_dataset import AnuraDataset

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        cuda.empty_cache()
        cuda.synchronize()
    gc.collect()
    print("GPU memory cleared")

def get_anura_datasets(config, data_dir, use_percentage=None):
    """
    Create train, validation, and test datasets from Anura data
    
    Args:
        config: Configuration object
        data_dir: Root directory containing Anura data
        use_percentage: Percentage of data to use (for quick testing)
    """
    datasets = {}
    
    # Load training data
    print("Loading Anura training data...")
    datasets["train"] = AnuraDataset(
        config=config,
        split="train",
        root_dir=data_dir,
        percentage=use_percentage
    )
    
    # Load validation data (used during training for early stopping)
    print("Loading Anura validation data...")
    datasets["valid"] = AnuraDataset(
        config=config,
        split="valid",
        root_dir=data_dir,
        percentage=use_percentage
    )
    
    # Load test data (used only for final evaluation)
    print("Loading Anura test data...")
    datasets["test"] = AnuraDataset(
        config=config,
        split="test",
        root_dir=data_dir,
        percentage=use_percentage
    )
    
    print(f"\nDataset splits created:")
    print(f"\tTrain: {len(datasets['train'])} samples")
    print(f"\tValid: {len(datasets['valid'])} samples (used during training)")
    print(f"\tTest: {len(datasets['test'])} samples (used for final evaluation)")
    print(f"\tNumber of species: {len(datasets['train'].label_columns)}")
    print("="*50)
    
    return datasets

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

def evaluate_model(model, eval_dataset, cfg_path, results_path, dataset_name="test", num_examples_to_print=5):
    """
    Evaluate the fine-tuned model on test/validation data
    
    Args:
        model: Fine-tuned model
        eval_dataset: Dataset to evaluate on
        cfg_path: Path to config file
        results_path: Path to save results
        dataset_name: Name of dataset being evaluated ("test" or "valid")
        num_examples_to_print: Number of example predictions to print
    
    Returns:
        dict: Evaluation results
    """
    results_file = os.path.join(results_path, f"fine_tune_{dataset_name}_results.txt")
    summary_file = os.path.join(results_path, f"fine_tune_{dataset_name}_summary.txt")
    results = []
    
    # Use all data for evaluation (no subset)
    val_indices = list(range(len(eval_dataset)))
    
    # Create evaluation dataframe
    eval_data = []
    for idx in val_indices:
        item = eval_dataset[idx]
        row = eval_dataset.df.iloc[idx]
        eval_data.append({
            'index': idx,
            'species_list': row[eval_dataset.label_columns].values.tolist(),
            'output': item['text'],
            'audio_path': row['audio_path'],
            'label_columns': eval_dataset.label_columns,
            'split': row.get('split', 'unknown')
        })
    
    eval_df = pd.DataFrame(eval_data)
    print(f"\nEvaluating on {dataset_name} set: {len(eval_df)} samples")
    
    if not os.path.exists(results_file):
        # Load the pipeline
        print("Loading evaluation pipeline...")
        model.eval()
        infer_pipe = Pipeline(model=model, cfg_path=cfg_path)
        
        # Run inference in batches to avoid memory issues
        print(f"Running inference on {len(eval_df)} samples...")
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
        
        # Save results
        with open(results_file, "w") as f:
            f.write("\n".join(results) + "\n")
        print(f"Results saved to: {results_file}")
    else:
        print(f"Loading cached results from {results_file}")
        with open(results_file) as f:
            for line in f:
                results.append(line.rstrip())
    
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
    
    # Create a mapping of normalized species names for lookup
    species_columns = eval_dataset.label_columns
    # Create a normalized lookup dictionary (case-insensitive, trimmed)
    species_lookup = {col.lower().strip(): col for col in species_columns}
    
    # Initialize metrics with original column names
    species_metrics = {col: {'tp': 0, 'fp': 0, 'fn': 0} for col in species_columns}
    
    # Evaluate predictions
    detailed_results = []
    total_correct_exact = 0
    total_correct_any = 0
    total_species_present = 0
    total_species_predicted = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for idx, (row, window_results) in enumerate(zip(eval_df.iterrows(), grouped_results)):
        row_data = row[1]
        ground_truth_text = row_data["output"].strip().lower()
        
        # Parse ground truth species with normalization
        if ground_truth_text == "none":
            ground_truth_species = set()
        else:
            # Split and clean each species name
            raw_species = [s.strip() for s in ground_truth_text.split(",")]
            ground_truth_species = set()
            for sp in raw_species:
                sp_norm = sp.lower().strip()
                if sp_norm in species_lookup:
                    ground_truth_species.add(species_lookup[sp_norm])
                else:
                    print(f"Warning: Unknown ground truth species '{sp}' not found in label columns")
        
        # Parse window predictions with normalization
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
                    if pred != "none":
                        # Try to match prediction to known species
                        pred_norm = pred.lower().strip()
                        if pred_norm in species_lookup:
                            matched_species = species_lookup[pred_norm]
                            if matched_species not in window_preds:
                                window_preds.append(matched_species)
                        else:
                            print(f"Warning: Unknown predicted species '{pred}' not found in label columns")
        
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
        
        # Per-species metrics - only for valid species
        for species in ground_truth_species:
            if species in species_metrics:  # Check if species exists in metrics
                if species in predicted_species:
                    species_metrics[species]['tp'] += 1
                else:
                    species_metrics[species]['fn'] += 1
        
        for species in predicted_species - set(["none"]):
            if species in species_metrics:  # Check if species exists in metrics
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
            print(f"Ground truth species (normalized): {ground_truth_species}")
            print(f"Window predictions: {window_preds}")
            print(f"Predicted species (normalized): {predicted_species - set(['none'])}")
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
    print(f"{dataset_name.upper()} SET EVALUATION SUMMARY")
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
        f.write(f"{dataset_name.upper()} SET EVALUATION SUMMARY\n")
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
    parser = argparse.ArgumentParser(description="Fine-tune NatureLM-audio on AnuraSet frog species classification")
    parser.add_argument("--naturelm_dir", type=str, default="NatureLMaudio", 
                       help="Location of the NatureLM-audio directory")
    parser.add_argument("--data_dir", type=str, default="data/AnuraSet", 
                       help="Location of the AnuraSet data directory")
    parser.add_argument("--cpu_offload", action="store_true", 
                       help="Enable CPU offloading")
    parser.add_argument("--output_dir", type=str, default="outputs/anura_finetune",
                       help="Custom output directory for results")
    parser.add_argument("--use_percentage", type=float, default=None,
                       help="Percentage of data to use (for quick testing)")
    parser.add_argument("--skip_test_eval", action="store_true",
                       help="Skip test set evaluation after training")
    parser.add_argument("--test_output", type=str, default="fine_tune_predictions.csv",
                       help="Output file for test predictions")
    args = parser.parse_args()

    # Load configuration
    if IN_COLAB:
        cfg_path = "/content/drive/MyDrive/NatureLMaudio/configs/finetune_anura.yaml"
    else:
        cfg_path = "NatureLMaudio/configs/finetune_anura.yaml"
    
    cfg = Config.from_sources(cfg_path)
    
    # Create job ID for the runner
    percentage_str = f"_pct{args.use_percentage}" if args.use_percentage else ""
    job_id = f"anura_finetune{percentage_str}_lora{cfg.model.lora_rank}_lr{cfg.run.optims.init_lr}"

    # Override output directory if specified
    out_path = os.path.join(args.output_dir, job_id)
    os.makedirs(out_path, exist_ok=True)
    
    # Load the base model
    print("Loading the model...")
    model, _ = load_model_and_config(cfg_path=cfg_path, device=cfg.model.device)
    
    # Clean up
    del _
    gc.collect()

    # Configure memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Configure LoRA
    model.lora = cfg.model.lora
    model.lora_rank = cfg.model.lora_rank
    model.lora_alpha = cfg.model.lora_alpha

    # Freeze non-LoRA parameters
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
            
            # Optionally offload to CPU
            if args.cpu_offload:
                param.data = param.data.cpu()
        else:
            param.requires_grad = True
            trainable_params += param.numel()
            # Ensure trainable params are on GPU
            if args.cpu_offload and param.device.type == 'cpu':
                param.data = param.data.cuda()
        total_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Prepare the datasets
    print("\nPreparing Anura datasets...")
    datasets = get_anura_datasets(cfg, args.data_dir, use_percentage=args.use_percentage)

    # Load the best model
    results_path = out_path
    best_model_path = os.path.join(results_path, "checkpoint_best.pth")

    if not os.path.exists(best_model_path):
        # Initialize the runner
        print("\nInitializing runner...")
        runner = Runner(cfg, model, datasets, job_id)

        # Start training
        print("\nStarting training...")
        runner.train()

        # Clear memory after training
        del runner
        gc.collect()
        clear_gpu_memory()

    # ================================================================================= #
    # Evaluation on test set
    if not args.skip_test_eval:
        print("\n" + "="*50)
        print("EVALUATING FINE-TUNED MODEL ON TEST SET")
        print("="*50)
        
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            
            # Load the base model first (if not already loaded)
            if 'model' not in locals() or model is None:
                model, _ = load_model_and_config(cfg_path=cfg_path, device=cfg.model.device)
            
            # Load checkpoint
            checkpoint = torch.load(best_model_path, map_location=cfg.model.device, weights_only=False)
            
            # For LoRA fine-tuning, load only the LoRA weights
            if 'model' in checkpoint:
                # Get the current model's state dict
                model_state = model.state_dict()
                
                # Update only the LoRA weights from checkpoint
                for name, param in checkpoint['model'].items():
                    if name in model_state:
                        model_state[name].copy_(param)
                    else:
                        print(f"Warning: Parameter {name} not found in model")
                
                # Load the updated state dict
                model.load_state_dict(model_state)
            else:
                # If checkpoint is just the state dict, try to load only matching keys
                model_state = model.state_dict()
                for name, param in checkpoint.items():
                    if name in model_state:
                        model_state[name].copy_(param)
                model.load_state_dict(model_state)
            
            print("Best model loaded successfully!")
            
            # Evaluate on test set
            eval_results = evaluate_model(
                model=model,
                eval_dataset=datasets["test"],
                cfg_path=cfg_path,
                results_path=results_path,
                dataset_name="test",
                num_examples_to_print=5
            )
            
            # Also save predictions CSV
            predictions_df = pd.DataFrame(eval_results['detailed_results'])
            predictions_file = os.path.join(results_path, args.test_output)
            predictions_df.to_csv(predictions_file, index=False)
            print(f"\nDetailed predictions saved to: {predictions_file}")
            
        else:
            print(f"Warning: Best model not found at {best_model_path}")
        
        clear_gpu_memory()

if __name__ == "__main__":
    main()
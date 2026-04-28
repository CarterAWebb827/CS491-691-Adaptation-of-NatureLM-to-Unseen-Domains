import os
import sys
import gc
from pathlib import Path
from huggingface_hub import login
import argparse
import pandas as pd
import torch
import torch.cuda as cuda
import json
import numpy as np

IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    current_dir = Path.cwd()
    naturelm_dir = Path(os.path.join(current_dir, "NatureLMaudio"))
    if str(naturelm_dir) not in sys.path:
        sys.path.insert(0, str(naturelm_dir))
    from NatureLM.config import Config
    from NatureLM.infer import load_model_and_config, Pipeline
    from NatureLM.runner import Runner
else:
    from NatureLMaudio.NatureLM.config import Config
    from NatureLMaudio.NatureLM.infer import load_model_and_config, Pipeline
    from NatureLMaudio.NatureLM.runner import Runner

# Try environment variable first, then fall back to interactive
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    print("Logging in with HF_TOKEN from environment...")
    login(token=hf_token)
else:
    print("No HF_TOKEN found in environment, will prompt for login...")
    login()

# Import Macaque dataset
from macaque_dataset import MacaqueDataset

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        cuda.empty_cache()
        cuda.synchronize()
    gc.collect()
    print("GPU memory cleared")

def get_macaque_datasets(config, data_dir, use_percentage=None, use_predefined_splits=True, valid_split_ratio=0.2, seed=42):
    """
    Create train, validation, and test datasets from Macaque data
    
    Args:
        config: Configuration object
        data_dir: Root directory containing Macaque data
        use_percentage: Percentage of data to use (for quick testing)
        use_predefined_splits: If True, use split column from metadata. If False, create random splits.
        valid_split_ratio: Ratio of training data to use for validation (when use_predefined_splits=True)
        seed: Random seed for reproducibility
    """
    datasets = {}
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load training data
    print("Loading Macaque training data...")
    datasets["train"] = MacaqueDataset(
        config=config,
        split="train",
        root_dir=data_dir,
        percentage=use_percentage,
        use_predefined_splits=use_predefined_splits,
        valid_split_ratio=valid_split_ratio,
        seed=seed
    )
    
    # Load validation data (used during training for early stopping)
    print("Loading Macaque validation data...")
    datasets["valid"] = MacaqueDataset(
        config=config,
        split="valid",
        root_dir=data_dir,
        percentage=use_percentage,
        use_predefined_splits=use_predefined_splits,
        valid_split_ratio=valid_split_ratio,
        seed=seed
    )
    
    # Load test data (used only for final evaluation)
    print("Loading Macaque test data...")
    datasets["test"] = MacaqueDataset(
        config=config,
        split="test",
        root_dir=data_dir,
        percentage=use_percentage,
        use_predefined_splits=use_predefined_splits,
        valid_split_ratio=valid_split_ratio,
        seed=seed
    )
    
    print(f"\nDataset splits created:")
    print(f"\tTrain: {len(datasets['train'])} samples")
    print(f"\tValid: {len(datasets['valid'])} samples (used during training)")
    print(f"\tTest: {len(datasets['test'])} samples (used for final evaluation)")
    print(f"\tNumber of call types: {len(datasets['train'].label_columns)}")
    print("="*50)
    
    return datasets

def evaluate_model(model, eval_dataset, cfg_path, results_path, dataset_name="test", num_examples_to_print=5):
    """
    Evaluate the fine-tuned model on test/validation data for Macaque call classification
    
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
            'call_type_list': row[eval_dataset.label_columns].values.tolist(),
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
            batch_instructions = ["<Audio><AudioHere></Audio> What type of macaque call is present in the audio, if any?"] * len(batch_paths)
            
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
    
    # Create a mapping of normalized call type names for lookup
    call_type_columns = eval_dataset.label_columns
    # Create a normalized lookup dictionary (case-insensitive, trimmed)
    call_type_lookup = {col.lower().strip(): col for col in call_type_columns}
    
    # Initialize metrics with original column names
    call_type_metrics = {col: {'tp': 0, 'fp': 0, 'fn': 0} for col in call_type_columns}
    
    # Evaluate predictions
    detailed_results = []
    total_correct_exact = 0
    total_correct_any = 0
    total_calls_present = 0
    total_calls_predicted = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for idx, (row, window_results) in enumerate(zip(eval_df.iterrows(), grouped_results)):
        row_data = row[1]
        ground_truth_text = row_data["output"].strip().lower()
        
        # Parse ground truth calls with normalization
        if ground_truth_text == "none":
            ground_truth_calls = set()
        else:
            # Split and clean each call type name
            raw_calls = [c.strip() for c in ground_truth_text.split(",")]
            ground_truth_calls = set()
            for ct in raw_calls:
                ct_norm = ct.lower().strip()
                if ct_norm in call_type_lookup:
                    ground_truth_calls.add(call_type_lookup[ct_norm])
                else:
                    print(f"Warning: Unknown ground truth call type '{ct}' not found in label columns")
        
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
                        # Try to match prediction to known call types
                        pred_norm = pred.lower().strip()
                        if pred_norm in call_type_lookup:
                            matched_call = call_type_lookup[pred_norm]
                            if matched_call not in window_preds:
                                window_preds.append(matched_call)
                        else:
                            print(f"Warning: Unknown predicted call type '{pred}' not found in label columns")
        
        # For multi-label, we'll consider any prediction
        predicted_calls = set(window_preds) if window_preds else set(["none"])
        
        # Calculate metrics
        exact_match = (predicted_calls == ground_truth_calls or 
                      (predicted_calls == set(["none"]) and ground_truth_calls == set()))
        if exact_match:
            total_correct_exact += 1
        
        # Any correct detection (if any predicted call type is in ground truth)
        if ground_truth_calls:
            any_correct = len(predicted_calls.intersection(ground_truth_calls)) > 0
            if any_correct:
                total_correct_any += 1
        
        # Per-call-type metrics - only for valid call types
        for call_type in ground_truth_calls:
            if call_type in call_type_metrics:  # Check if call type exists in metrics
                if call_type in predicted_calls:
                    call_type_metrics[call_type]['tp'] += 1
                else:
                    call_type_metrics[call_type]['fn'] += 1
        
        for call_type in predicted_calls - set(["none"]):
            if call_type in call_type_metrics:  # Check if call type exists in metrics
                if call_type not in ground_truth_calls:
                    call_type_metrics[call_type]['fp'] += 1
        
        tp = len(predicted_calls.intersection(ground_truth_calls))
        fp = len(predicted_calls - ground_truth_calls - set(["none"]))
        fn = len(ground_truth_calls - predicted_calls)
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        total_calls_present += len(ground_truth_calls)
        total_calls_predicted += len(predicted_calls - set(["none"]))
        
        detailed_results.append({
            'index': idx,
            'ground_truth': ground_truth_text,
            'ground_truth_set': list(ground_truth_calls),
            'window_predictions': window_preds,
            'predicted_set': list(predicted_calls - set(["none"])),
            'exact_match': exact_match,
            'any_correct': any_correct if ground_truth_calls else None,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        })
        
        if idx < num_examples_to_print:
            print(f"\n{'='*50}")
            print(f"Example {idx}:")
            print(f"Ground truth: {ground_truth_text}")
            print(f"Ground truth call types (normalized): {ground_truth_calls}")
            print(f"Window predictions: {window_preds}")
            print(f"Predicted call types (normalized): {predicted_calls - set(['none'])}")
            print(f"Exact match: {exact_match}")
    
    # Calculate overall metrics
    exact_accuracy = (total_correct_exact / len(eval_df)) * 100
    any_accuracy = (total_correct_any / len(eval_df)) * 100 if total_calls_present > 0 else 0
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-call-type metrics
    per_call_type_results = {}
    for call_type, metrics in call_type_metrics.items():
        ct_precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
        ct_recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
        ct_f1 = 2 * (ct_precision * ct_recall) / (ct_precision + ct_recall) if (ct_precision + ct_recall) > 0 else 0
        
        per_call_type_results[call_type] = {
            'true_positives': metrics['tp'],
            'false_positives': metrics['fp'],
            'false_negatives': metrics['fn'],
            'precision': ct_precision,
            'recall': ct_recall,
            'f1': ct_f1
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
        
        f.write("Per-call-type Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Call Type':<30} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
        f.write("-" * 80 + "\n")
        
        for call_type, metrics in sorted(per_call_type_results.items()):
            f.write(f"{call_type:<30} {metrics['true_positives']:<6} {metrics['false_positives']:<6} "
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
        'per_call_type_metrics': per_call_type_results,
        'detailed_results': detailed_results
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NatureLM-audio on Macaque call type classification")
    parser.add_argument("--naturelm_dir", type=str, default="NatureLMaudio", 
                       help="Location of the NatureLM-audio directory")
    parser.add_argument("--data_dir", type=str, default="data/macaques", 
                       help="Location of the Macaque data directory")
    parser.add_argument("--valid_split_ratio", type=float, default=0.2,
                   help="Ratio of training data to use for validation (when use_predefined_splits=True)")
    parser.add_argument("--cpu_offload", action="store_true", 
                       help="Enable CPU offloading")
    parser.add_argument("--output_dir", type=str, default="outputs/macaque_finetune",
                       help="Custom output directory for results")
    parser.add_argument("--use_percentage", type=float, default=None,
                       help="Percentage of data to use (for quick testing)")
    parser.add_argument("--skip_test_eval", action="store_true",
                       help="Skip test set evaluation after training")
    parser.add_argument("--test_output", type=str, default="fine_tune_predictions.csv",
                       help="Output file for test predictions")
    
    # Hyperparameter arguments
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Override learning rate in config")
    parser.add_argument("--warmup_steps", type=int, default=None,
                       help="Override warmup steps in config")
    parser.add_argument("--weight_decay", type=float, default=None,
                       help="Override weight decay in config")
    parser.add_argument("--max_epochs", type=int, default=None,
                       help="Override max epochs in config")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size in config")
    parser.add_argument("--lora_rank", type=int, default=None,
                       help="Override LoRA rank in config")
    parser.add_argument("--lora_alpha", type=int, default=None,
                       help="Override LoRA alpha in config")
    parser.add_argument("--accum_grad_iters", type=int, default=None,
                       help="Override gradient accumulation steps in config")
    
    # QLoRA-specific arguments
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Enable 4-bit quantization for QLoRA")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Compute dtype for 4-bit layers")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                       choices=["fp4", "nf4"],
                       help="Quantization type (fp4 or nf4)")
    parser.add_argument("--use_nested_quant", action="store_true",
                       help="Enable nested quantization for more memory savings")
    
    # Gradient checkpointing arguments
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--gradient_checkpointing_kwargs", type=str, default='{"use_reentrant": false}',
                       help="JSON string of gradient checkpointing kwargs")
    parser.add_argument("--clear_cache_every_n_steps", type=int, default=10,
                       help="Clear CUDA cache every N steps")
    
    parser.add_argument("--use_grid_output", action="store_true",
                       help="Use output_dir directly without appending job_id")
    parser.add_argument("--use_predefined_splits", action="store_true", default=True,
                       help="Use predefined splits from metadata (ensures consistency across runs)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Determine config path based on environment
    if IN_COLAB:
        cfg_path = "/content/drive/MyDrive/NatureLMaudio/configs/finetune.yaml"
    else:
        # Look for config in standard locations
        possible_paths = [
            "NatureLMaudio/configs/finetune.yaml",
            "configs/finetune.yaml",
            os.path.join(args.naturelm_dir, "configs/finetune.yaml")
        ]
        cfg_path = None
        for path in possible_paths:
            if os.path.exists(path):
                cfg_path = path
                break
        
        if cfg_path is None:
            print("Warning: finetune.yaml not found, using finetune_anura.yaml as template")
            for path in possible_paths:
                alt_path = path.replace("macaque", "anura")
                if os.path.exists(alt_path):
                    cfg_path = alt_path
                    break
    
    if cfg_path is None or not os.path.exists(cfg_path):
        raise FileNotFoundError("Could not find a valid config file. Please create finetune.yaml")
    
    print(f"Using config: {cfg_path}")
    cfg = Config.from_sources(cfg_path)
    cfg.run.output_dir = args.output_dir
    
    # Override hyperparameters if provided
    if args.learning_rate is not None:
        cfg.run.optims.init_lr = args.learning_rate
        print(f"Overriding learning rate to: {args.learning_rate}")
    
    if args.warmup_steps is not None:
        cfg.run.optims.warmup_steps = args.warmup_steps
        print(f"Overriding warmup steps to: {args.warmup_steps}")
    
    if args.weight_decay is not None:
        cfg.run.optims.weight_decay = args.weight_decay
        print(f"Overriding weight decay to: {args.weight_decay}")
    
    if args.max_epochs is not None:
        cfg.run.optims.max_epoch = args.max_epochs
        print(f"Overriding max epochs to: {args.max_epochs}")
    
    if args.batch_size is not None:
        cfg.run.batch_size_train = args.batch_size
        cfg.run.batch_size_eval = args.batch_size
        print(f"Overriding batch size to: {args.batch_size}")
    
    if args.lora_rank is not None:
        cfg.model.lora_rank = args.lora_rank
        print(f"Overriding LoRA rank to: {args.lora_rank}")
    
    if args.lora_alpha is not None:
        cfg.model.lora_alpha = args.lora_alpha
        print(f"Overriding LoRA alpha to: {args.lora_alpha}")
    
    if args.accum_grad_iters is not None:
        cfg.run.accum_grad_iters = args.accum_grad_iters
        print(f"Overriding gradient accumulation steps to: {args.accum_grad_iters}")
    
    # Add QLoRA parameters to config
    if not hasattr(cfg.model, 'use_4bit'):
        cfg.model.use_4bit = args.use_4bit
    if not hasattr(cfg.model, 'bnb_4bit_compute_dtype'):
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        cfg.model.bnb_4bit_compute_dtype = dtype_map[args.bnb_4bit_compute_dtype]
    if not hasattr(cfg.model, 'bnb_4bit_quant_type'):
        cfg.model.bnb_4bit_quant_type = args.bnb_4bit_quant_type
    if not hasattr(cfg.model, 'use_nested_quant'):
        cfg.model.use_nested_quant = args.use_nested_quant
    
    # Parse gradient checkpointing kwargs
    gradient_checkpointing_kwargs = json.loads(args.gradient_checkpointing_kwargs)
    
    # Update config with gradient checkpointing settings
    cfg.model.use_gradient_checkpointing = args.use_gradient_checkpointing
    cfg.model.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs
    cfg.run.enable_gradient_checkpointing = args.use_gradient_checkpointing
    cfg.run.clear_cache_every_n_steps = args.clear_cache_every_n_steps
    
    # Create job ID for logging purposes
    percentage_str = f"_pct{args.use_percentage}" if args.use_percentage else ""
    job_id = f"macaque_finetune{percentage_str}_lora{cfg.model.lora_rank}_lr{cfg.run.optims.init_lr}"
    
    # Use the output directory directly
    out_path = args.output_dir
    os.makedirs(out_path, exist_ok=True)
    
    print(f"Output directory: {out_path}")
    print(f"Job ID (for reference): {job_id}")
    
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
    
    # Verify trainable parameters
    print("\n" + "="*50)
    print("VERIFYING TRAINABLE PARAMETERS")
    print("="*50)
    
    trainable_params = 0
    total_params = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora" in name.lower():
                lora_params += param.numel()
                print(f"\t - Trainable LoRA param: {name} - shape: {param.shape}")
            else:
                print(f"\t - Trainable non-LoRA param: {name} - shape: {param.shape}")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"LoRA parameters: {lora_params:,}")
    
    if trainable_params == 0:
        print("ERROR: No trainable parameters found!")
        print("Please check LoRA configuration.")
        sys.exit(1)
    elif lora_params == 0 and trainable_params > 0:
        print("WARNING: Trainable parameters are not LoRA parameters!")
    
    # Prepare the datasets
    print("\nPreparing Macaque datasets...")
    datasets = get_macaque_datasets(cfg, args.data_dir, use_percentage=args.use_percentage, 
                                    use_predefined_splits=args.use_predefined_splits, 
                                    valid_split_ratio=args.valid_split_ratio, seed=args.random_seed)
    
    # Check for best model
    results_path = os.path.join(out_path, "macaque_finetune_lora" + str(cfg.model.lora_rank) + "_lr" + str(cfg.run.optims.init_lr))
    best_model_path = os.path.join(results_path, "checkpoint_best.pth")
    
    if not os.path.exists(best_model_path):
        # Initialize the runner
        print("\nInitializing runner...")
        runner = Runner(cfg, model, datasets, job_id, use_grid_output=args.use_grid_output)
        
        # Start training
        print("\nStarting training...")
        runner.train()
        
        # Clear memory after training
        del runner
        gc.collect()
        clear_gpu_memory()
    else:
        print(f"Found existing checkpoint at {best_model_path}, skipping training...")
    
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
                model_state = model.state_dict()
                for name, param in checkpoint['model'].items():
                    if name in model_state:
                        model_state[name].copy_(param)
                    else:
                        print(f"Warning: Parameter {name} not found in model")
                model.load_state_dict(model_state)
            else:
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
            
            predictions_df = pd.DataFrame(eval_results['detailed_results'])
            predictions_file = os.path.join(results_path, args.test_output)
            predictions_df.to_csv(predictions_file, index=False)
            print(f"\nDetailed predictions saved to: {predictions_file}")
            
        else:
            print(f"Warning: Best model not found at {best_model_path}")
        
        clear_gpu_memory()
    
    print("\n" + "="*50)
    print("MACAQUE FINE-TUNING COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
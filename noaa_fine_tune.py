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
    from NatureLMaudio.NatureLM.config import Config
    from NatureLMaudio.NatureLM.infer import load_model_and_config, Pipeline
    from NatureLMaudio.NatureLM.runner import Runner
else:
    from NatureLMaudio.NatureLM.config import Config
    from NatureLMaudio.NatureLM.infer import load_model_and_config, Pipeline
    from NatureLMaudio.NatureLM.runner import Runner

# Login to HuggingFace
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    print("Logging in with HF_TOKEN from environment...")
    login(token=hf_token)
else:
    print("No HF_TOKEN found in environment, will prompt for login...")
    login()

# Import NOAA dataset
from noaa_dataset import RightWhaleDataset

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        cuda.empty_cache()
        cuda.synchronize()
    gc.collect()
    print("GPU memory cleared")

def get_noaa_datasets(config, data_dir, use_percentage=None, seed=42):
    """
    Create train, validation, and test datasets from NOAA Right Whale data
    
    Args:
        config: Configuration object
        data_dir: Root directory containing NOAA data
        use_percentage: Percentage of data to use (for quick testing)
        seed: Random seed for reproducibility
    """
    datasets = {}
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load training data
    print("Loading NOAA training data...")
    datasets["train"] = RightWhaleDataset(
        config=config,
        split="train",
        root_dir=data_dir,
        percentage=use_percentage
    )
    
    # Load validation data (used during training for early stopping)
    print("Loading NOAA validation data...")
    datasets["valid"] = RightWhaleDataset(
        config=config,
        split="valid",
        root_dir=data_dir,
        percentage=use_percentage
    )
    
    # Load test data (used only for final evaluation)
    print("Loading NOAA test data...")
    datasets["test"] = RightWhaleDataset(
        config=config,
        split="test",
        root_dir=data_dir,
        percentage=use_percentage
    )
    
    print(f"\nDataset splits created:")
    print(f"\tTrain: {len(datasets['train'])} samples")
    print(f"\tValid: {len(datasets['valid'])} samples (used during training)")
    print(f"\tTest: {len(datasets['test'])} samples (used for final evaluation)")
    print(f"\tPositive chunks in train: {datasets['train'].df[RightWhaleDataset.SPECIES_NAME].sum()}")
    print(f"\tPositive chunks in valid: {datasets['valid'].df[RightWhaleDataset.SPECIES_NAME].sum()}")
    print(f"\tPositive chunks in test: {datasets['test'].df[RightWhaleDataset.SPECIES_NAME].sum()}")
    print("="*50)
    
    return datasets

def evaluate_model(model, eval_dataset, cfg_path, results_path, dataset_name="test", num_examples_to_print=5):
    """
    Evaluate the fine-tuned model on test/validation data for Right Whale detection
    
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
    
    # Use all data for evaluation
    eval_indices = list(range(len(eval_dataset)))
    
    print(f"\nEvaluating on {dataset_name} set: {len(eval_indices)} samples")
    
    # Check if we have cached results
    if os.path.exists(results_file):
        print(f"Loading cached results from {results_file}")
        with open(results_file, 'r') as f:
            results = [line.rstrip() for line in f]
    else:
        # Load the pipeline
        print("Loading evaluation pipeline...")
        model.eval()
        infer_pipe = Pipeline(model=model, cfg_path=cfg_path)
        
        # Prepare audio clips and instructions
        print(f"Preparing audio clips for {len(eval_dataset)} samples...")
        clip_audio = []
        instructions = []
        
        for idx in eval_indices:
            row = eval_dataset.df.iloc[idx]
            audio = eval_dataset.load_audio(
                row["audio_path"], 
                start_time=row["clip_start_seconds"], 
                end_time=row["clip_end_seconds"]
            )
            clip_audio.append(audio.numpy())
            instructions.append(row["instruction"])
        
        # Run inference
        print(f"Running inference on {len(clip_audio)} samples...")
        results = infer_pipe(clip_audio, instructions)
        
        # Save results
        with open(results_file, "w") as f:
            f.write("\n".join(results) + "\n")
        print(f"Results saved to: {results_file}")
    
    # Initialize metrics
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    detailed_results = []
    
    # Define helper function for right whale detection
    def is_right_whale(prediction):
        prediction = prediction.strip().lower()
        right_whale_indicators = [
            "right whale", "north atlantic right whale", "north pacific right whale",
            "northern right whale", "southern right whale",
            "eubalaena glacialis", "eubalaena japonica", "eubalaena australis"
        ]
        return any(indicator in prediction for indicator in right_whale_indicators)
    
    for idx, result in enumerate(results):
        row = eval_dataset.df.iloc[idx]
        actual_right_whale = bool(row[RightWhaleDataset.SPECIES_NAME])
        
        # Parse prediction
        if result and ":" in result:
            prediction = result.split(":", 1)[1].strip()
        else:
            prediction = result.strip() if result else ""
        
        predicted_right_whale = is_right_whale(prediction)
        
        # Update confusion matrix
        if predicted_right_whale and actual_right_whale:
            tp += 1
            confusion = "TP"
        elif predicted_right_whale and not actual_right_whale:
            fp += 1
            confusion = "FP"
        elif not predicted_right_whale and actual_right_whale:
            fn += 1
            confusion = "FN"
        else:
            tn += 1
            confusion = "TN"
        
        detailed_results.append({
            'index': idx,
            'audio_path': row["audio_path"],
            'chunk_start': row["clip_start_seconds"],
            'chunk_end': row["clip_end_seconds"],
            'actual_right_whale': actual_right_whale,
            'predicted_right_whale': predicted_right_whale,
            'raw_prediction': prediction,
            'confusion': confusion
        })
        
        if idx < num_examples_to_print:
            print(f"\n{'='*50}")
            print(f"Example {idx}:")
            print(f"Audio: {Path(row['audio_path']).name}")
            print(f"Chunk: {row['clip_start_seconds']:.2f}s - {row['clip_end_seconds']:.2f}s")
            print(f"Actual right whale: {actual_right_whale}")
            print(f"Raw prediction: {prediction[:200]}...")
            print(f"Predicted right whale: {predicted_right_whale}")
            print(f"Confusion: {confusion}")
    
    # Calculate metrics
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    positive_chunks = int(eval_dataset.df[RightWhaleDataset.SPECIES_NAME].sum())
    negative_chunks = len(eval_dataset) - positive_chunks
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"{dataset_name.upper()} SET EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {len(eval_dataset)}")
    print(f"Positive chunks: {positive_chunks}")
    print(f"Negative chunks: {negative_chunks}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}")
    print(f"  FP: {fp}")
    print(f"  TN: {tn}")
    print(f"  FN: {fn}")
    print(f"\nMetrics:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Save summary to file
    with open(summary_file, "w") as f:
        f.write(f"{dataset_name.upper()} SET EVALUATION SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Total samples: {len(eval_dataset)}\n")
        f.write(f"Positive chunks: {positive_chunks}\n")
        f.write(f"Negative chunks: {negative_chunks}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {tp}\n")
        f.write(f"  FP: {fp}\n")
        f.write(f"  TN: {tn}\n")
        f.write(f"  FN: {fn}\n\n")
        
        f.write("Metrics:\n")
        f.write(f"  Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")
    
    print(f"\nEvaluation summary saved to: {summary_file}")
    
    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_file = os.path.join(results_path, f"fine_tune_{dataset_name}_detailed.csv")
    detailed_df.to_csv(detailed_file, index=False)
    print(f"Detailed results saved to: {detailed_file}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'total_samples': len(eval_dataset),
        'positive_chunks': positive_chunks,
        'negative_chunks': negative_chunks,
        'detailed_results': detailed_results
    }

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NatureLM-audio on NOAA Right Whale dataset")
    parser.add_argument("--naturelm_dir", type=str, default="NatureLMaudio", 
                       help="Location of the NatureLM-audio directory")
    parser.add_argument("--data_dir", type=str, default="/content/drive/MyDrive/RightWhaleData", 
                       help="Location of the NOAA Right Whale data directory")
    parser.add_argument("--cpu_offload", action="store_true", 
                       help="Enable CPU offloading")
    parser.add_argument("--output_dir", type=str, default="outputs/noaa_finetune",
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
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Determine config path based on environment
    if IN_COLAB:
        cfg_path = "/content/drive/MyDrive/NatureLMaudio/configs/finetune_noaa.yaml"
    else:
        # Look for config in standard locations
        possible_paths = [
            "NatureLMaudio/configs/finetune_noaa.yaml",
            "configs/finetune_noaa.yaml",
            os.path.join(args.naturelm_dir, "configs/finetune_noaa.yaml")
        ]
        cfg_path = None
        for path in possible_paths:
            if os.path.exists(path):
                cfg_path = path
                break
        
        if cfg_path is None:
            print("Warning: finetune_noaa.yaml not found, using finetune_anura.yaml as template")
            for path in possible_paths:
                alt_path = path.replace("noaa", "anura")
                if os.path.exists(alt_path):
                    cfg_path = alt_path
                    break
    
    if cfg_path is None or not os.path.exists(cfg_path):
        raise FileNotFoundError("Could not find a valid config file. Please create finetune_noaa.yaml")
    
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
    job_id = f"noaa_finetune{percentage_str}_lora{cfg.model.lora_rank}_lr{cfg.run.optims.init_lr}"
    
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
    print("\nPreparing NOAA datasets...")
    datasets = get_noaa_datasets(cfg, args.data_dir, use_percentage=args.use_percentage, seed=args.random_seed)
    
    # Check for best model
    results_path = out_path
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
            
            # Also evaluate on validation set
            print("\n" + "="*50)
            print("EVALUATING ON VALIDATION SET")
            print("="*50)
            
            eval_results_valid = evaluate_model(
                model=model,
                eval_dataset=datasets["valid"],
                cfg_path=cfg_path,
                results_path=results_path,
                dataset_name="valid",
                num_examples_to_print=3
            )
            
            # Save combined metrics
            metrics_summary = {
                'test': eval_results,
                'valid': eval_results_valid
            }
            metrics_file = os.path.join(results_path, "fine_tune_metrics_summary.json")
            with open(metrics_file, 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                def convert(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                json.dump(metrics_summary, f, indent=2, default=convert)
            print(f"\nMetrics summary saved to: {metrics_file}")
            
        else:
            print(f"Warning: Best model not found at {best_model_path}")
        
        clear_gpu_memory()
    
    print("\n" + "="*50)
    print("NOAA FINE-TUNING COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
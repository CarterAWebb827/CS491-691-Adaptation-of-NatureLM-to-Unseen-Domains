import os
import sys
import gc
from pathlib import Path
from huggingface_hub import login
import argparse
import pandas as pd
import torch
from torch.utils.data import random_split

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

# Import our RFCx dataset
from rfcx_dataset import RFCFrogDataset, RFCTestDataset

def get_rfcx_datasets(config, data_dir, use_fp=False, val_split=0.1):
    """
    Create train and validation datasets from RFCx data
    
    Args:
        config: Configuration object
        data_dir: Root directory containing RFCx data
        use_fp: Whether to include false positives in training
        val_split: Fraction of training data to use for validation
    """
    datasets = {}
    
    # Load training data (TP only or TP+FP)
    print(f"Loading training data (use_fp={use_fp})...")
    full_train_dataset = RFCFrogDataset(
        config=config, 
        split="train_all" if use_fp else "train", 
        root_dir=data_dir,
        use_fp=use_fp
    )
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    datasets["train"], datasets["valid"] = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Store reference to original dataset for label access
    datasets["train"].dataset = full_train_dataset
    datasets["valid"].dataset = full_train_dataset
    
    # Test dataset (just recording IDs for submission)
    datasets["test"] = RFCTestDataset(config=config, root_dir=data_dir)
    
    print(f"\nDataset splits created:")
    print(f"\tTrain: {len(datasets['train'])} samples")
    print(f"\tValid: {len(datasets['valid'])} samples")
    print(f"\tTest: {len(datasets['test'])} recording IDs")
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

def evaluate_model(model, eval_dataset, cfg_path, results_path, num_examples_to_print=5):
    """
    Evaluate the fine-tuned model on validation data
    
    Args:
        model: Fine-tuned model
        eval_dataset: Validation dataset
        cfg_path: Path to config file
        results_path: Path to save results
        num_examples_to_print: Number of example predictions to print
    
    Returns:
        dict: Evaluation results
    """
    results_file = os.path.join(results_path, "validation_results.txt")
    results = []
    
    # Get a subset of validation data for evaluation
    # Since we don't have labeled test data, we'll use a portion of validation
    val_size = min(500)
    val_indices = torch.randperm(len(eval_dataset))[:val_size]
    
    # Create evaluation dataframe
    eval_data = []
    for idx in val_indices:
        item = eval_dataset[idx]
        if hasattr(eval_dataset, 'dataset'):
            # For Subset dataset, need to get original index
            orig_idx = eval_dataset.indices[idx]
            row = eval_dataset.dataset.df.iloc[orig_idx]
            eval_data.append({
                'recording_id': row['recording_id'],
                'species_id': row['species_id'] if 'species_id' in row else None,
                'output': row['output'],
                'audio_path': row['audio_path']
            })
    
    eval_df = pd.DataFrame(eval_data)
    
    if not os.path.exists(results_file):
        # Load the pipeline
        print("Loading evaluation pipeline...")
        model.eval()
        infer_pipe = Pipeline(model=model, cfg_path=cfg_path)
        
        # Run inference
        print(f"Running inference on {len(eval_df)} validation samples...")
        results = infer_pipe(eval_df["audio_path"], 
                           ["<Audio><AudioHere></Audio> What is the scientific name for the frog species in the audio, if any?"] * len(eval_df))
        
        # Save results
        with open(results_file, "w") as f:
            f.write("\n".join(results) + "\n")
        print(f"Results saved to: {results_file}")
    else:
        print(f"Loading cached results from {results_file}")
        with open(results_file) as f:
            for line in f:
                results.append(line.rstrip())
    
    # Group results
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
    
    # Evaluate predictions
    detailed_results = []
    total_correct = 0
    
    species_mapping = RFCFrogDataset.SPECIES_MAPPING
    id_to_name = {k: v[0].lower() for k, v in species_mapping.items()}
    name_to_id = {v[0].lower(): k for k, v in species_mapping.items()}
    
    for idx, (row, window_results) in enumerate(zip(eval_df.iterrows(), grouped_results)):
        row_data = row[1]
        ground_truth = row_data["output"].strip().lower()
        species_id = row_data["species_id"] if pd.notna(row_data["species_id"]) else None
        
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
            print(f"Species ID: {species_id}")
            print(f"Ground truth: {ground_truth}")
            print(f"Window predictions: {window_preds}")
            print(f"Aggregated prediction: {most_common_pred}")
            print(f"Correct: {is_correct}")
    
    # Calculate accuracy
    accuracy = (total_correct / len(eval_df)) * 100
    
    # Calculate per-species accuracy
    species_accuracy = {}
    for species_id in range(11):
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
    print("VALIDATION EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {len(eval_df)}")
    print(f"Correct predictions: {total_correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if species_accuracy:
        print(f"\nPer-species Accuracy:")
        for species_id, stats in species_accuracy.items():
            print(f"  {stats['species_code']} ({stats['species_name']}): "
                  f"{stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    
    return {
        'accuracy': accuracy,
        'total_samples': len(eval_df),
        'correct_predictions': total_correct,
        'detailed_results': detailed_results,
        'species_accuracy': species_accuracy
    }

def create_test_submission(model, test_dataset, cfg_path, output_file, batch_size=8):
    """
    Create submission file for test set
    """
    from NatureLMaudio.NatureLM.infer import Pipeline
    
    print("Creating test submission...")
    
    # Load sample submission template
    submission_template = pd.read_csv(Path(test_dataset.root_dir) / "sample_submission.csv")
    submission_df = submission_template.copy()
    
    # Set all species columns to 0 initially
    species_cols = [f's{i}' for i in range(24)]
    submission_df[species_cols] = 0
    
    # Load pipeline
    model.eval()
    infer_pipe = Pipeline(model=model, cfg_path=cfg_path)
    
    # Group by recording_id
    unique_recordings = submission_df['recording_id'].unique()
    
    print(f"Processing {len(unique_recordings)} test recordings...")
    
    for i, recording_id in enumerate(unique_recordings):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(unique_recordings)}")
        
        audio_path = test_dataset.root_dir / "test" / f"{recording_id}.flac"
        
        # Get model prediction
        instruction = "<Audio><AudioHere></Audio> What are the scientific name(s) for the species in the audio, if any?"
        results = infer_pipe([str(audio_path)], [instruction])
        
        # Parse prediction
        prediction_text = results[0].lower() if results else "none"
        
        # Map prediction to species probabilities
        for species_id in range(11):
            species_name = RFCFrogDataset.SPECIES_MAPPING[species_id][0].lower()
            if species_name in prediction_text:
                submission_df.loc[submission_df['recording_id'] == recording_id, f's{species_id}'] = 1.0
    
    # Save submission
    submission_df.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    
    return submission_df

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NatureLM-audio on RFCx frog species classification")
    parser.add_argument("--naturelm_dir", type=str, default="NatureLMaudio", 
                       help="Location of the NatureLM-audio directory")
    parser.add_argument("--data_dir", type=str, default="data/RFCx", 
                       help="Location of the RFCx data directory")
    parser.add_argument("--use_fp", action="store_true", 
                       help="Include false positives in training")
    parser.add_argument("--cpu_offload", action="store_true", 
                       help="Enable CPU offloading")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory for results")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Fraction of training data to use for validation")
    parser.add_argument("--skip_eval", action="store_true",
                       help="Skip validation evaluation")
    parser.add_argument("--create_submission", action="store_true",
                       help="Create submission file after training")
    parser.add_argument("--submission_file", type=str, default="submission.csv",
                       help="Output file for submission")
    args = parser.parse_args()

    # Load configuration
    if IN_COLAB:
        cfg_path = "/content/drive/MyDrive/NatureLMaudio/configs/finetune.yaml"
    else:
        cfg_path = "NatureLMaudio/configs/finetune.yaml"
    
    cfg = Config.from_sources(cfg_path)
    
    # Override output directory if specified
    if args.output_dir:
        cfg["run"]["output_dir"] = args.output_dir
    
    # Create job ID for the runner
    fp_str = "_with_fp" if args.use_fp else ""
    job_id = f"rfcx_finetune{fp_str}_lora{cfg.model.lora_rank}_lr{cfg.run.optims.init_lr}"
    
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
    print("\nPreparing RFCx datasets...")
    datasets = get_rfcx_datasets(cfg, args.data_dir, use_fp=args.use_fp, val_split=args.val_split)

    # Initialize the runner
    print("\nInitializing runner...")
    runner = Runner(cfg, model, datasets, job_id)

    # Start training
    print("\nStarting training...")
    runner.train()

    # ================================================================================= #
    # Evaluation on validation set
    if not args.skip_eval:
        print("\n" + "="*50)
        print("EVALUATING FINE-TUNED MODEL ON VALIDATION SET")
        print("="*50)
        
        # Load the best model
        results_path = cfg["run"]["output_dir"]
        best_model_path = os.path.join(results_path, "checkpoint_best.pth")
        
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=cfg.model.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Best model loaded successfully!")
            
            # Evaluate on validation set
            eval_results = evaluate_model(
                model=model,
                eval_dataset=datasets["valid"],
                cfg_path=cfg_path,
                results_path=results_path,
                num_examples_to_print=5
            )
            
            # Save evaluation summary
            summary_file = os.path.join(results_path, "evaluation_summary.txt")
            with open(summary_file, "w") as f:
                f.write("FINE-TUNED MODEL EVALUATION ON RFCx VALIDATION SET\n")
                f.write("="*50 + "\n")
                f.write(f"Overall Accuracy: {eval_results['accuracy']:.2f}%\n")
                f.write(f"Total Samples: {eval_results['total_samples']}\n")
                f.write(f"Correct Predictions: {eval_results['correct_predictions']}\n\n")
                
                f.write("Per-species Accuracy:\n")
                for species_id, stats in eval_results['species_accuracy'].items():
                    f.write(f"  {stats['species_code']} ({stats['species_name']}): "
                           f"{stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})\n")
            
            print(f"\nEvaluation summary saved to: {summary_file}")
            
        else:
            print(f"Warning: Best model not found at {best_model_path}")
    
    # Create submission file if requested
    if args.create_submission:
        print("\n" + "="*50)
        print("CREATING TEST SUBMISSION")
        print("="*50)
        
        # Load best model for submission
        results_path = cfg["run"]["output_dir"]
        best_model_path = os.path.join(results_path, "checkpoint_best.pth")
        
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=cfg.model.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            submission_df = create_test_submission(
                model=model,
                test_dataset=datasets["test"],
                cfg_path=cfg_path,
                output_file=os.path.join(results_path, args.submission_file)
            )
        else:
            print("Best model not found, using current model for submission")
            submission_df = create_test_submission(
                model=model,
                test_dataset=datasets["test"],
                cfg_path=cfg_path,
                output_file=os.path.join(results_path, args.submission_file)
            )

if __name__ == "__main__":
    main()
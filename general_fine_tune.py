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
from torch.utils.data import Dataset, ConcatDataset
import random

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

# Import datasets
from macaque_dataset import MacaqueDataset
from noaa_dataset import RightWhaleDataset
from anura_dataset import AnuraDataset
from NatureLMaudio.NatureLM.dataset import collater

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        cuda.empty_cache()
        cuda.synchronize()
    gc.collect()
    print("GPU memory cleared")

class CombinedDataset(Dataset):
    """
    Dataset that combines multiple datasets
    """
    def __init__(self, datasets, dataset_names, config):
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.config = config
        self.collater = collater
        self.dataset_indices = []
        
        # Create mapping from combined index to (dataset_idx, sample_idx)
        for dataset_idx, dataset in enumerate(datasets):
            for sample_idx in range(len(dataset)):
                self.dataset_indices.append((dataset_idx, sample_idx))
        
        print(f"Combined dataset created with {len(self.dataset_indices)} total samples")
        for name, dataset in zip(dataset_names, datasets):
            print(f"  {name}: {len(dataset)} samples")
        
    def __len__(self):
        return len(self.dataset_indices)
    
    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.dataset_indices[idx]
        item = self.datasets[dataset_idx][sample_idx]
        
        # Ensure raw_wav is a proper 1D tensor
        if 'raw_wav' in item:
            raw_wav = item['raw_wav']
            # Handle different tensor shapes
            if isinstance(raw_wav, torch.Tensor):
                if raw_wav.ndim == 0:
                    # 0-d tensor - convert to 1D
                    raw_wav = raw_wav.unsqueeze(0)
                elif raw_wav.ndim > 1:
                    # Multi-dimensional - squeeze to 1D
                    raw_wav = raw_wav.reshape(-1)
            item['raw_wav'] = raw_wav
        
        return item

def get_combined_datasets(config, data_dirs, use_percentage=None, macaque_percentage=None, 
                         noaa_percentage=None, anura_percentage=None, valid_split_ratio=0.2, seed=42):
    """
    Create combined train, validation, and test datasets using the original dataset classes
    
    Args:
        use_percentage: Default percentage for all datasets (0-100)
        macaque_percentage: Specific percentage for Macaque dataset (0-100)
        noaa_percentage: Specific percentage for NOAA dataset (0-100)
        anura_percentage: Specific percentage for Anura dataset (0-100)
    """
    
    print("\n" + "="*70)
    print("CREATING COMBINED DATASETS FROM MACAQUE, NOAA, AND ANURA")
    print("="*70)
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Determine percentages for each dataset
    # If specific percentage is provided, use it; otherwise use default; otherwise use None (full dataset)
    macaque_pct = macaque_percentage if macaque_percentage is not None else use_percentage
    noaa_pct = noaa_percentage if noaa_percentage is not None else use_percentage
    anura_pct = anura_percentage if anura_percentage is not None else use_percentage
    
    print(f"Dataset percentages: Macaque={macaque_pct if macaque_pct else '100'}%, "
          f"NOAA={noaa_pct if noaa_pct else '100'}%, "
          f"Anura={anura_pct if anura_pct else '100'}%")
    
    # Load original datasets for each type
    all_train_datasets = []
    all_valid_datasets = []
    all_test_datasets = []
    dataset_names = []
    
    # Load Macaque datasets
    if "macaque" in data_dirs:
        print("\nLoading Macaque datasets...")
        try:
            macaque_train = MacaqueDataset(
                config=config,
                split="train",
                root_dir=data_dirs["macaque"],
                use_predefined_splits=True,
                valid_split_ratio=valid_split_ratio,
                seed=seed
            )
            macaque_valid = MacaqueDataset(
                config=config,
                split="valid",
                root_dir=data_dirs["macaque"],
                use_predefined_splits=True,
                valid_split_ratio=valid_split_ratio,
                seed=seed
            )
            macaque_test = MacaqueDataset(
                config=config,
                split="test",
                root_dir=data_dirs["macaque"],
                use_predefined_splits=True,
                valid_split_ratio=valid_split_ratio,
                seed=seed
            )
            
            all_train_datasets.append(macaque_train)
            all_valid_datasets.append(macaque_valid)
            all_test_datasets.append(macaque_test)
            dataset_names.append("macaque")
            
            print(f"  Macaque - Train: {len(macaque_train)}, Valid: {len(macaque_valid)}, Test: {len(macaque_test)}")
        except Exception as e:
            print(f"  Error loading Macaque dataset: {e}")
    
    # Load NOAA datasets
    if "noaa" in data_dirs:
        print("\nLoading NOAA datasets...")
        try:
            noaa_train = RightWhaleDataset(
                config=config,
                split="train",
                root_dir=data_dirs["noaa"],
                percentage=noaa_pct
            )
            noaa_valid = RightWhaleDataset(
                config=config,
                split="valid",
                root_dir=data_dirs["noaa"],
                percentage=noaa_pct
            )
            noaa_test = RightWhaleDataset(
                config=config,
                split="test",
                root_dir=data_dirs["noaa"],
                percentage=noaa_pct
            )
            
            all_train_datasets.append(noaa_train)
            all_valid_datasets.append(noaa_valid)
            all_test_datasets.append(noaa_test)
            dataset_names.append("noaa")
            
            print(f"  NOAA - Train: {len(noaa_train)}, Valid: {len(noaa_valid)}, Test: {len(noaa_test)}")
        except Exception as e:
            print(f"  Error loading NOAA dataset: {e}")
    
    # Load Anura datasets
    if "anura" in data_dirs:
        print("\nLoading Anura datasets...")
        try:
            anura_train = AnuraDataset(
                config=config,
                split="train",
                root_dir=data_dirs["anura"],
                percentage=anura_pct,
                use_predefined_splits=True
            )
            anura_valid = AnuraDataset(
                config=config,
                split="valid",
                root_dir=data_dirs["anura"],
                percentage=anura_pct,
                use_predefined_splits=True
            )
            anura_test = AnuraDataset(
                config=config,
                split="test",
                root_dir=data_dirs["anura"],
                percentage=anura_pct,
                use_predefined_splits=True
            )
            
            all_train_datasets.append(anura_train)
            all_valid_datasets.append(anura_valid)
            all_test_datasets.append(anura_test)
            dataset_names.append("anura")
            
            print(f"  Anura - Train: {len(anura_train)}, Valid: {len(anura_valid)}, Test: {len(anura_test)}")
        except Exception as e:
            print(f"  Error loading Anura dataset: {e}")
    
    # Create combined datasets
    print("\n" + "="*70)
    print("CREATING COMBINED DATASETS")
    print("="*70)
    
    combined_datasets = {}
    
    if all_train_datasets:
        combined_datasets["train"] = CombinedDataset(
            all_train_datasets, dataset_names, config
        )
        print(f"Train set: {len(combined_datasets['train'])} total samples")
    
    if all_valid_datasets:
        combined_datasets["valid"] = CombinedDataset(
            all_valid_datasets, dataset_names, config
        )
        print(f"Valid set: {len(combined_datasets['valid'])} total samples")
    
    if all_test_datasets:
        combined_datasets["test"] = CombinedDataset(
            all_test_datasets, dataset_names, config
        )
        print(f"Test set: {len(combined_datasets['test'])} total samples")
    
    print("\n" + "="*70)
    print("DATASET CREATION COMPLETE")
    print("="*70)
    
    return combined_datasets

def evaluate_model(model, eval_dataset, cfg_path, results_path, dataset_name="test", num_examples_to_print=5, use_full_test=False, data_dirs=None):
    """
    Evaluation with detailed per-dataset metrics including precision, recall, and F1
    
    Args:
        use_full_test: If True, evaluate on all available test data regardless of training subset
        data_dirs: Dictionary mapping dataset names to their data directories
    """
    results_file = os.path.join(results_path, f"combined_fine_tune_{dataset_name}_results.json")
    summary_file = os.path.join(results_path, f"combined_fine_tune_{dataset_name}_summary.txt")
    detailed_metrics_file = os.path.join(results_path, f"combined_fine_tune_{dataset_name}_detailed_metrics.csv")
    
    # If using full test, we need to load full datasets
    if use_full_test:
        if data_dirs is None:
            print("Warning: data_dirs not provided for full test evaluation. Using existing test data.")
        else:
            print("\n" + "="*70)
            print("FULL TEST EVALUATION MODE")
            print("Loading complete test datasets...")
            print("="*70)
            
            # Load full test datasets without percentage restriction
            full_datasets = []
            full_dataset_names = []
            
            for dataset_idx, dataset_name in enumerate(eval_dataset.dataset_names):
                try:
                    if dataset_name == "macaque":
                        from macaque_dataset import MacaqueDataset
                        full_dataset = MacaqueDataset(
                            config=eval_dataset.config,
                            split="test",
                            root_dir=data_dirs["macaque"],
                            use_predefined_splits=True,
                            valid_split_ratio=0.2,
                            seed=42
                        )
                        full_datasets.append(full_dataset)
                        full_dataset_names.append("macaque")
                        print(f"  Loaded full Macaque test: {len(full_dataset)} samples")
                        
                    elif dataset_name == "noaa":
                        from noaa_dataset import RightWhaleDataset
                        full_dataset = RightWhaleDataset(
                            config=eval_dataset.config,
                            split="test",
                            root_dir=data_dirs["noaa"],
                            percentage=None
                        )
                        full_datasets.append(full_dataset)
                        full_dataset_names.append("noaa")
                        print(f"  Loaded full NOAA test: {len(full_dataset)} samples")
                        
                    elif dataset_name == "anura":
                        from anura_dataset import AnuraDataset
                        full_dataset = AnuraDataset(
                            config=eval_dataset.config,
                            split="test",
                            root_dir=data_dirs["anura"],
                            percentage=None,
                            use_predefined_splits=True
                        )
                        full_datasets.append(full_dataset)
                        full_dataset_names.append("anura")
                        print(f"  Loaded full Anura test: {len(full_dataset)} samples")
                        
                except Exception as e:
                    print(f"  Error loading full {dataset_name} test dataset: {e}")
                    # Fall back to the original subset
                    full_datasets.append(eval_dataset.datasets[dataset_idx])
                    full_dataset_names.append(dataset_name)
                    print(f"  Using existing {dataset_name} test subset: {len(eval_dataset.datasets[dataset_idx])} samples")
            
            # Create new combined dataset with full test data
            if full_datasets:
                eval_dataset = CombinedDataset(full_datasets, full_dataset_names, eval_dataset.config)
    
    print(f"\nEvaluating on {dataset_name} set: {len(eval_dataset)} samples")
    
    if not os.path.exists(results_file):
        print("Loading evaluation pipeline...")
        model.eval()
        infer_pipe = Pipeline(model=model, cfg_path=cfg_path)
        
        # Prepare data for evaluation with more metadata
        evaluation_data = []
        
        for idx in range(len(eval_dataset)):
            if idx % 1000 == 0:
                print(f"  Preparing sample {idx}/{len(eval_dataset)}")
                
            dataset_idx, sample_idx = eval_dataset.dataset_indices[idx]
            dataset = eval_dataset.datasets[dataset_idx]
            dataset_type = eval_dataset.dataset_names[dataset_idx]
            
            # Get sample data
            try:
                item = dataset[sample_idx]
            except Exception as e:
                print(f"  Error loading item {idx} from {dataset_type}: {e}")
                continue
            
            # Handle different dataset types
            try:
                if dataset_type == "macaque":
                    audio_path = dataset.df.iloc[sample_idx]['audio_path']
                    instruction = "<Audio><AudioHere></Audio> What type of macaque call is present in the audio, if any?"
                    ground_truth = item['text']
                    label_columns = dataset.label_columns
                    
                elif dataset_type == "noaa":
                    # For NOAA, use the original approach from noaa_fine_tune.py
                    row = dataset.df.iloc[sample_idx]
                    # Load the specific audio chunk using the dataset's load_audio method
                    audio = dataset.load_audio(
                        row["audio_path"], 
                        start_time=row["clip_start_seconds"], 
                        end_time=row["clip_end_seconds"]
                    )
                    audio_path = audio.numpy() if hasattr(audio, 'numpy') else audio
                    instruction = row["instruction"]
                    ground_truth = item['text']
                    label_columns = ['Right Whale']
                    
                elif dataset_type == "anura":
                    audio_path = dataset.df.iloc[sample_idx]['audio_path']
                    instruction = "<Audio><AudioHere></Audio> What are the scientific name(s) for the species in the audio, if any?"
                    ground_truth = item['text']
                    label_columns = dataset.label_columns
                
                evaluation_data.append({
                    'idx': len(evaluation_data), # Re-index for evaluation
                    'dataset_type': dataset_type,
                    'audio': audio_path,
                    'instruction': instruction,
                    'ground_truth': ground_truth,
                    'label_columns': label_columns,
                    'sample_idx': sample_idx
                })
                
            except Exception as e:
                print(f"  Error processing sample {idx} from {dataset_type}: {e}")
                continue
        
        # Run inference in batches
        print(f"Running inference on {len(evaluation_data)} samples...")
        batch_size = 8
        all_results = []
        
        for i in range(0, len(evaluation_data), batch_size):
            if i % 100 == 0 or i >= len(evaluation_data) - batch_size:
                print(f"  Processed {i}/{len(evaluation_data)}")
            
            batch_end = min(i + batch_size, len(evaluation_data))
            batch_audio = [d['audio'] for d in evaluation_data[i:batch_end]]
            batch_instructions = [d['instruction'] for d in evaluation_data[i:batch_end]]
            
            try:
                batch_results = infer_pipe(batch_audio, batch_instructions)
                all_results.extend(batch_results)
            except Exception as e:
                print(f"  Error processing batch {i}-{batch_end}: {e}")
                # Add empty results for failed batch
                all_results.extend([""] * len(batch_audio))
        
        # Combine results with metadata
        for i, result in enumerate(all_results):
            if i < len(evaluation_data):
                evaluation_data[i]['prediction'] = result
        
        # Save results
        with open(results_file, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = []
            for item in evaluation_data:
                serializable_item = item.copy()
                if isinstance(serializable_item.get('audio'), np.ndarray):
                    serializable_item['audio'] = serializable_item['audio'].tolist()
                serializable_data.append(serializable_item)
            
            json.dump(serializable_data, f, indent=2)
        print(f"Results saved to: {results_file}")
        
    else:
        print(f"Loading cached results from {results_file}")
        with open(results_file) as f:
            evaluation_data = json.load(f)
    
    # Evaluate each dataset type with appropriate metrics
    all_metrics = {}
    all_detailed_results = []
    
    for dataset_type in ["macaque", "noaa", "anura"]:
        dataset_results = [r for r in evaluation_data if r['dataset_type'] == dataset_type]
        
        if not dataset_results:
            print(f"No results found for {dataset_type}")
            continue
        
        print(f"\n{'='*70}")
        print(f"{dataset_type.upper()} DATASET RESULTS")
        print(f"{'='*70}")
        
        if dataset_type == "macaque":
            metrics = evaluate_macaque_results(dataset_results)
        elif dataset_type == "noaa":
            metrics = evaluate_noaa_results(dataset_results)
        elif dataset_type == "anura":
            metrics = evaluate_anura_results(dataset_results)
        
        all_metrics[dataset_type] = metrics
        
        # Print dataset-specific metrics
        print_metrics_summary(dataset_type, metrics, num_examples_to_print)
        
        # Add dataset type to detailed results
        for result in metrics['detailed_results']:
            result['dataset_type'] = dataset_type
            all_detailed_results.append(result)
    
    # Calculate overall metrics (weighted average)
    if all_metrics:
        print(f"\n{'='*70}")
        print(f"OVERALL COMBINED RESULTS")
        print(f"{'='*70}")
        
        overall_metrics = calculate_overall_metrics(all_metrics)
        print_overall_summary(overall_metrics)
        
        # Save comprehensive summary
        save_comprehensive_summary(summary_file, all_metrics, overall_metrics)
    else:
        print("No metrics to calculate. All evaluations failed.")
        overall_metrics = {'total_samples': 0}
    
    # Save detailed results to CSV
    if all_detailed_results:
        detailed_df = pd.DataFrame(all_detailed_results)
        detailed_df.to_csv(detailed_metrics_file, index=False)
        print(f"\nDetailed metrics saved to: {detailed_metrics_file}")
    
    return {
        'overall_metrics': overall_metrics,
        'per_dataset_metrics': all_metrics
    }

def evaluate_macaque_results(results):
    """Evaluate Macaque call classification results"""
    # Similar to the logic in macaque_fine_tune.py
    correct_exact = 0
    correct_any = 0
    total = len(results)
    
    detailed_results = []
    
    for result in results:
        ground_truth = result['ground_truth'].strip().lower()
        prediction = result['prediction'].strip().lower()
        
        # Parse prediction (handle "filename: prediction" format)
        if ":" in prediction:
            prediction = prediction.split(":", 1)[1].strip()
        
        # Check exact match
        exact_match = (ground_truth == prediction)
        if exact_match:
            correct_exact += 1
        
        # Check any correct (for multi-label)
        gt_calls = set(ground_truth.split(",")) if ground_truth != "none" else set()
        pred_calls = set(prediction.split(",")) if prediction != "none" else set()
        
        any_correct = len(gt_calls & pred_calls) > 0 if gt_calls else (prediction == "none")
        if any_correct:
            correct_any += 1
        
        detailed_results.append({
            'ground_truth': ground_truth,
            'prediction': prediction,
            'exact_match': exact_match,
            'any_correct': any_correct
        })
    
    return {
        'total_samples': total,
        'exact_accuracy': (correct_exact / total) * 100,
        'any_accuracy': (correct_any / total) * 100,
        'correct_exact': correct_exact,
        'correct_any': correct_any,
        'detailed_results': detailed_results
    }

def evaluate_noaa_results(results):
    """Evaluate NOAA Right Whale detection results"""
    tp = fp = tn = fn = 0
    detailed_results = []
    
    def is_right_whale(prediction):
        prediction = prediction.strip().lower()
        indicators = ["right whale", "eubalaena"]
        return any(ind in prediction for ind in indicators)
    
    for result in results:
        ground_truth = result['ground_truth'].strip().lower()
        prediction = result['prediction'].strip().lower()
        
        if ":" in prediction:
            prediction = prediction.split(":", 1)[1].strip()
        
        actual_positive = ("right whale" in ground_truth or "eubalaena" in ground_truth)
        predicted_positive = is_right_whale(prediction)
        
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1
        
        detailed_results.append({
            'ground_truth': ground_truth,
            'prediction': prediction,
            'actual_positive': actual_positive,
            'predicted_positive': predicted_positive
        })
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_samples': total,
        'accuracy': accuracy * 100,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'detailed_results': detailed_results
    }

def evaluate_anura_results(results):
    """Evaluate Anura species classification results"""
    correct_exact = 0
    correct_any = 0
    total = len(results)
    
    detailed_results = []
    
    for result in results:
        ground_truth = result['ground_truth'].strip().lower()
        prediction = result['prediction'].strip().lower()
        
        if ":" in prediction:
            prediction = prediction.split(":", 1)[1].strip()
        
        exact_match = (ground_truth == prediction)
        if exact_match:
            correct_exact += 1
        
        gt_species = set(ground_truth.split(",")) if ground_truth != "none" else set()
        pred_species = set(prediction.split(",")) if prediction != "none" else set()
        
        any_correct = len(gt_species & pred_species) > 0 if gt_species else (prediction == "none")
        if any_correct:
            correct_any += 1
        
        detailed_results.append({
            'ground_truth': ground_truth,
            'prediction': prediction,
            'exact_match': exact_match,
            'any_correct': any_correct
        })
    
    return {
        'total_samples': total,
        'exact_accuracy': (correct_exact / total) * 100,
        'any_accuracy': (correct_any / total) * 100,
        'correct_exact': correct_exact,
        'correct_any': correct_any,
        'detailed_results': detailed_results
    }

def print_metrics_summary(dataset_type, metrics, num_examples):
    """Print formatted metrics summary for a dataset"""
    print(f"\nDataset: {dataset_type.upper()}")
    print(f"Total samples: {metrics['total_samples']}")
    
    if dataset_type in ["macaque", "anura"]:
        print(f"Exact match accuracy: {metrics['exact_accuracy']:.2f}%")
        print(f"Any correct detection accuracy: {metrics['any_accuracy']:.2f}%")
        print(f"Correct (exact): {metrics['correct_exact']}/{metrics['total_samples']}")
        print(f"Correct (any): {metrics['correct_any']}/{metrics['total_samples']}")
    elif dataset_type == "noaa":
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}")
        print(f"  FN: {metrics['fn']}, TN: {metrics['tn']}")
    
    # Print examples
    print(f"\nExample predictions:")
    for i in range(min(num_examples, len(metrics['detailed_results']))):
        result = metrics['detailed_results'][i]
        print(f"  {i+1}. GT: {result['ground_truth'][:50]}...")
        print(f"     Pred: {result['prediction'][:50]}...")

def calculate_overall_metrics(all_metrics):
    """Calculate weighted overall metrics across all datasets"""
    total_samples = sum(m['total_samples'] for m in all_metrics.values())
    
    overall = {'total_samples': total_samples}
    
    # Weighted accuracy
    if all(m.get('accuracy') is not None for m in all_metrics.values()):
        weighted_acc = sum(m['accuracy'] * m['total_samples'] for m in all_metrics.values()) / total_samples
        overall['weighted_accuracy'] = weighted_acc
    
    # For exact match accuracy
    exact_match_datasets = [m for m in all_metrics.values() if 'exact_accuracy' in m]
    if exact_match_datasets:
        exact_samples = sum(m['total_samples'] for m in exact_match_datasets)
        if exact_samples > 0:
            weighted_exact = sum(m['exact_accuracy'] * m['total_samples'] for m in exact_match_datasets) / exact_samples
            overall['weighted_exact_accuracy'] = weighted_exact
    
    return overall

def print_overall_summary(overall_metrics):
    """Print overall summary across all datasets"""
    print(f"Total samples across all datasets: {overall_metrics['total_samples']}")
    if 'weighted_accuracy' in overall_metrics:
        print(f"Weighted accuracy: {overall_metrics['weighted_accuracy']:.2f}%")
    if 'weighted_exact_accuracy' in overall_metrics:
        print(f"Weighted exact match accuracy: {overall_metrics['weighted_exact_accuracy']:.2f}%")

def save_comprehensive_summary(summary_file, all_metrics, overall_metrics):
    """Save comprehensive summary to file"""
    with open(summary_file, "w") as f:
        f.write("="*70 + "\n")
        f.write("COMBINED DATASET EVALUATION - COMPREHENSIVE SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS (Weighted by dataset size):\n")
        f.write("-"*50 + "\n")
        for key, value in overall_metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.2f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Per-dataset metrics
        for dataset_type, metrics in all_metrics.items():
            f.write("="*50 + "\n")
            f.write(f"{dataset_type.upper()} DATASET\n")
            f.write("="*50 + "\n")
            
            for key, value in metrics.items():
                if key != 'detailed_results':
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n" if value < 1 else f"  {key}: {value:.2f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NatureLM-audio on combined Macaque, NOAA, and Anura datasets")
    parser.add_argument("--naturelm_dir", type=str, default="NatureLMaudio", 
                       help="Location of the NatureLM-audio directory")
    parser.add_argument("--macaque_data_dir", type=str, default="data/macaques", 
                       help="Location of the Macaque data directory")
    parser.add_argument("--noaa_data_dir", type=str, default="data/NOAA", 
                       help="Location of the NOAA Right Whale data directory")
    parser.add_argument("--anura_data_dir", type=str, default="data/AnuraSet", 
                       help="Location of the AnuraSet data directory")
    parser.add_argument("--valid_split_ratio", type=float, default=0.2,
                       help="Ratio of data to use for validation and testing")
    parser.add_argument("--cpu_offload", action="store_true", 
                       help="Enable CPU offloading")
    parser.add_argument("--output_dir", type=str, default="outputs/combined_finetune",
                       help="Custom output directory for results")
    parser.add_argument("--use_percentage", type=float, default=None,
                    help="Default percentage of data to use for all datasets (0-100)")
    parser.add_argument("--macaque_percentage", type=float, default=None,
                    help="Percentage of Macaque data to use (0-100)")
    parser.add_argument("--noaa_percentage", type=float, default=None,
                    help="Percentage of NOAA data to use (0-100)")
    parser.add_argument("--anura_percentage", type=float, default=None,
                    help="Percentage of Anura data to use (0-100)")
    parser.add_argument("--skip_test_eval", action="store_true",
                       help="Skip test set evaluation after training")
    parser.add_argument("--full_test", action="store_true",
                       help="Evaluate on all test data, not just the training subset")
    
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
    
    # Determine config path
    if IN_COLAB:
        cfg_path = "/content/drive/MyDrive/NatureLMaudio/configs/finetune.yaml"
    else:
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
            raise FileNotFoundError("Could not find a valid config file. Please create finetune.yaml")
    
    print(f"Using config: {cfg_path}")
    cfg = Config.from_sources(cfg_path)
    cfg.run.output_dir = args.output_dir
    
    # Override hyperparameters if provided
    if args.learning_rate is not None:
        cfg.run.optims.init_lr = args.learning_rate
    if args.warmup_steps is not None:
        cfg.run.optims.warmup_steps = args.warmup_steps
    if args.weight_decay is not None:
        cfg.run.optims.weight_decay = args.weight_decay
    if args.max_epochs is not None:
        cfg.run.optims.max_epoch = args.max_epochs
    if args.batch_size is not None:
        cfg.run.batch_size_train = args.batch_size
        cfg.run.batch_size_eval = args.batch_size
    if args.lora_rank is not None:
        cfg.model.lora_rank = args.lora_rank
    if args.lora_alpha is not None:
        cfg.model.lora_alpha = args.lora_alpha
    if args.accum_grad_iters is not None:
        cfg.run.accum_grad_iters = args.accum_grad_iters
    
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
    
    # Create job ID
    percentage_str = f"_pct{args.use_percentage}" if args.use_percentage else ""
    job_id = f"combined_finetune{percentage_str}_lora{cfg.model.lora_rank}_lr{cfg.run.optims.init_lr}"
    
    # Create output directory
    out_path = args.output_dir
    os.makedirs(out_path, exist_ok=True)
    
    print(f"Output directory: {out_path}")
    print(f"Job ID: {job_id}")
    
    # Load the base model
    print("\nLoading the model...")
    model, _ = load_model_and_config(cfg_path=cfg_path, device=cfg.model.device)
    
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
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"LoRA parameters: {lora_params:,}")
    
    if trainable_params == 0:
        print("ERROR: No trainable parameters found!")
        sys.exit(1)
    
    # Prepare the balanced datasets
    data_dirs = {
        "macaque": args.macaque_data_dir,
        "noaa": args.noaa_data_dir,
        "anura": args.anura_data_dir
    }
    
    print("\nPreparing balanced combined datasets...")
    datasets = get_combined_datasets(
        cfg, data_dirs,
        use_percentage=args.use_percentage,
        macaque_percentage=args.macaque_percentage,
        noaa_percentage=args.noaa_percentage,
        anura_percentage=args.anura_percentage,
        valid_split_ratio=args.valid_split_ratio,
        seed=args.random_seed
    )
    
    # Check for best model
    results_path = os.path.join(out_path, f"combined_finetune_lora{cfg.model.lora_rank}_lr{cfg.run.optims.init_lr}")
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
        print("EVALUATING FINE-TUNED MODEL ON COMBINED TEST SET")
        if args.full_test:
            print("USING FULL TEST DATASETS (not just training subset)")
        print("="*50)
        
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            
            if 'model' not in locals() or model is None:
                model, _ = load_model_and_config(cfg_path=cfg_path, device=cfg.model.device)
            
            checkpoint = torch.load(best_model_path, map_location=cfg.model.device, weights_only=False)
            
            if 'model' in checkpoint:
                model_state = model.state_dict()
                for name, param in checkpoint['model'].items():
                    if name in model_state:
                        model_state[name].copy_(param)
                model.load_state_dict(model_state)
            else:
                model.load_state_dict(checkpoint)
            
            print("Best model loaded successfully!")
            
            # Evaluate on test set with full test option
            eval_results = evaluate_model(
                model=model,
                eval_dataset=datasets["test"],
                cfg_path=cfg_path,
                results_path=results_path,
                dataset_name="test",
                num_examples_to_print=5,
                use_full_test=args.full_test,
                data_dirs=data_dirs
            )
            
        else:
            print(f"Warning: Best model not found at {best_model_path}")
        
        clear_gpu_memory()
    
    print("\n" + "="*50)
    print("COMBINED FINE-TUNING COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()
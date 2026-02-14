import pandas as pd
import numpy as np
from pathlib import Path
import torch
import gc
from torch.utils.data import Dataset
import pickle

current_dir = Path.cwd()

class XenoDataset(Dataset):
    def __init__(self, parquet_dir):
        self.parquet_dir = parquet_dir
        self.cache_dir = parquet_dir / "xeno_canto_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Get all the pickle files in the directory
        self.parquet_files = sorted(list(self.parquet_dir.glob("*.parquet")))
        print(f"Found {len(self.parquet_files)} parquet files in {self.parquet_dir}")

        # Check if we have a cached index file
        index_cache_path = self.cache_dir / "file_indices.pkl"
        if index_cache_path.exists():
            print("Loading cached file indices...")
            with open(index_cache_path, "rb") as f:
                cache_data = pickle.load(f)
                self.file_indices = cache_data["file_indices"]
                self.total_samples = cache_data["total_samples"]
                self.cached_files = cache_data.get("cached_files", [])
        else:
            self.file_indices = [] # List of (file_idx, indices) for Xeno-Canto call-type rows
            self.total_samples = 0
            self.cached_files = []
            
            print("Scanning and caching Xeno-Canto call-type data...")
            self._process_and_cache_all_files()
            
            # Save the index cache
            cache_data = {
                "file_indices": self.file_indices,
                "total_samples": self.total_samples,
                "cached_files": self.cached_files
            }
            with open(index_cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"Saved index cache to {index_cache_path}")
        
        # Initialize current file tracking
        self.current_file_idx = -1
        self.current_file_data = None
        
        # Load the full DataFrame of all Xeno-Canto call-type rows
        print("Loading full Xeno-Canto call-type DataFrame...")
        self.full_df = self._load_df()
        print(f"Loaded full DataFrame with {len(self.full_df)} samples")
    
    def _process_and_cache_all_files(self):
        for file_idx, file_path in enumerate(self.parquet_files):
            cache_file = self.cache_dir / f"xeno_canto_file_{file_idx}.pkl"

            # Read in only the source_dataset and dataset_name columns
            df_filter = pd.read_parquet(file_path, columns=["source_dataset", "dataset_name"])

            # Only store the wanted source and datasets
            xc_ct_mask = (df_filter["source_dataset"].str.lower() == "xeno-canto") & (df_filter["dataset_name"].str.lower() == "call-type")
            xc_ct_indices = df_filter[xc_ct_mask].index.to_list()
            if xc_ct_indices:
                # Load the filtered rows and cache them
                df_full = pd.read_parquet(file_path)
                filtered_df = df_full.iloc[xc_ct_indices].reset_index(drop=True)

                filtered_df.to_pickle(cache_file)

                # Clean up
                del df_full, filtered_df
                gc.collect()

                self.cached_files.append(cache_file)

                self.file_indices.append({
                    "file_idx": file_idx,
                    "indices": xc_ct_indices,
                    "start_idx": self.total_samples,
                    "end_idx": self.total_samples + len(xc_ct_indices),
                    "cache_path": cache_file
                })
                self.total_samples += len(xc_ct_indices)

                print(f"\tFile {file_idx+1}/{len(self.parquet_files)}: {len(xc_ct_indices)} Xeno-Canto call-type samples")

    def _load_df(self):
        all_dfs = []

        for file_info in self.file_indices:
            cache_path = file_info.get("cache_path")
            df = pd.read_pickle(cache_path)
            all_dfs.append(df)
        
        full_df = pd.concat(all_dfs, ignore_index=True)

        return full_df

    def _load_file(self, file_index):
        """Load a specified parquet file, filtering for xeno-canto"""
        if self.current_file_idx == file_index:
            return # The file is already loaded
        
        for info in self.file_indices:
            if info["file_idx"] == file_index:
                file_info = info
            else:
                file_info = None
        
        cache_path = file_info.get("cache_path")
        self.current_file_data = pd.read_pickle(cache_path)
        self.current_file_idx = file_index
    
    def _get_file_and_local_index(self, global_index):
        for file_info in self.file_indices:
            if file_info["start_idx"] <= global_index < file_info["end_idx"]:
                local_idx = global_index - file_info["start_idx"]
                return file_info["file_idx"], local_idx
        raise IndexError(f"Global index {global_index} out of range")

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, index):
        # Find which file contains the given index
        file_idx, local_idx = self._get_file_and_local_index(index)

        # Load the correct file if not already loaded
        if file_idx != self.current_file_idx:
            self._load_file(file_idx)
        
        # Get the row
        row = self.current_file_data.iloc[local_idx]

        # Extract the audio data
        audio_str = row["audio"]

        # Clean the string and convert to numpy array
        audio_str = audio_str[1:-1].strip() # Remove the brackets from the string
        audio_vals = np.fromstring(audio_str.split(), dtype=np.float32, sep=" ") # Split by whitespace and save as floats
        audio = torch.tensor(audio_vals, dtype=torch.float32)

        # Get the prompt
        prompt = row.get("instruction_text", row.get("instruction", ""))

        # Get the ground truth output
        output = row["output"]

        # Get the ID
        recording_id = row["id"]

        # Get the metadata
        metadata = row["metadata"]

        return {
            "raw_wav": [audio], # Wrap in list as expected by processor
            "text": str(output), # Ground truth label (call or song)
            "prompt": prompt,
            "task": "call-type-classification",
            "id": str(recording_id),
            "dataset_name": "xeno-canto-call-type",
            "source_dataset": "xeno-canto",
            "metadata": metadata,
            "index": index
        }
    
    def get_full_df(self):
        return self.full_df
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import soundfile as sf
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T

# Determine if running on cluster
ON_VISTA = os.path.exists('/home1') or 'TACC' in os.environ.get('HOSTNAME', '')

if ON_VISTA:
    # On Vista, use WORK directory
    WORK_DIR = os.environ.get('WORK', '/work')
    NATURELM_DIR = os.path.join(WORK_DIR, 'NatureLMaudio')
    
    # Add to path
    if str(NATURELM_DIR) not in sys.path:
        sys.path.insert(0, str(NATURELM_DIR))
        print(f"Added {NATURELM_DIR} to Python path")
    
    from NatureLM.dataset import collater
else:
    # Colab/local path logic
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    if IN_COLAB:
        current_dir = Path.cwd()
        naturelm_dir = current_dir / "NatureLMaudio"
        if str(naturelm_dir) not in sys.path:
            sys.path.insert(0, str(naturelm_dir))
        from NatureLM.dataset import collater
    else:
        from NatureLMaudio.NatureLM.dataset import collater

current_dir = Path.cwd()
anura_dir = Path(os.path.join(current_dir, "data/AnuraSet"))

class AnuraDataset(Dataset):
    # Class variables to store splits across instances of the AnuraDataset class
    _train_df = None
    _valid_df = None
    _test_df = None
    _full_df = None
    _label_columns = None
    _is_prepared = False
    _reduced_df = None  # Store reduced version for percentage testing

    def __init__(self, config, percentage=None, split="train", root_dir="data/AnuraSet", use_predefined_splits=True):
        """
        Args:
            config: Configuration object
            percentage: Percentage of data to use (for quick testing)
            split: Which split to load ("train", "valid", or "test")
            root_dir: Root directory containing Anura data
            use_predefined_splits: If True, use split column from metadata. If False, create random splits.
        """
        self.config = config
        self.percentage = percentage
        self.split = split
        self.root_dir = Path(root_dir)
        self.sample_rate = 16000
        self.max_length_samples = 10 * self.sample_rate
        self.audio_column = "fname"
        self.station_column = "site"
        self.collater = collater
        self.use_predefined_splits = use_predefined_splits

        # Prepare the metadata (only once)
        if not AnuraDataset._is_prepared:
            self._prepare_metadata()

        # Apply percentage reduction to the FULL dataset if specified
        # This ensures we get a representative subset across ALL splits
        if self.percentage is not None and AnuraDataset._reduced_df is None:
            self._create_reduced_dataset()
        
        # Load the appropriate split
        self._load_split()

        self.label_columns = AnuraDataset._label_columns

        print(f"Loaded {self.split} split: {len(self.df)} samples")
        print(f"Number of species: {len(self.label_columns)}")
    
    def _create_reduced_dataset(self):
        """Create a reduced version of the entire dataset by taking a percentage of each split"""
        print(f"\n{'='*50}")
        print(f"Creating reduced dataset: {self.percentage*100}% of full data")
        print(f"{'='*50}")
        
        # Get the full dataframe
        full_df = AnuraDataset._full_df
        
        # Create a reduced version by sampling from each split proportionally
        reduced_dfs = []
        
        for split_name in ['train', 'valid', 'test']:
            split_df = full_df[full_df['split'] == split_name].copy()
            
            if len(split_df) > 0:
                # Calculate how many samples to take from this split
                n_samples = max(1, int(len(split_df) * self.percentage))
                
                # Stratify by species presence to maintain distribution
                stratify_labels = split_df[AnuraDataset._label_columns].sum(axis=1) > 0
                
                # Sample with fixed random state for reproducibility
                sampled_df = split_df.groupby(stratify_labels, group_keys=False).apply(
                    lambda x: x.sample(
                        n=min(len(x), max(1, int(len(x) * self.percentage))),
                        random_state=42
                    )
                )
                
                # Alternative: use train_test_split for more control
                # _, sampled_df = train_test_split(
                #     split_df,
                #     train_size=n_samples,
                #     random_state=42,
                #     stratify=stratify_labels
                # )
                
                reduced_dfs.append(sampled_df)
                print(f"  {split_name}: {len(split_df)} -> {len(sampled_df)} samples ({len(sampled_df)/len(split_df)*100:.1f}%)")
        
        # Combine all reduced splits
        AnuraDataset._reduced_df = pd.concat(reduced_dfs, axis=0)
        print(f"Total reduced dataset: {len(AnuraDataset._reduced_df)} samples")
        print(f"{'='*50}\n")
    
    def _load_split(self):
        """Load the appropriate split from either full or reduced dataset"""
        # Choose which dataframe to use
        if self.percentage is not None:
            source_df = AnuraDataset._reduced_df
        else:
            source_df = AnuraDataset._full_df
        
        # Filter by split
        if self.split == "train":
            self.df = source_df[source_df['split'] == 'train'].copy()
        elif self.split == "valid":
            self.df = source_df[source_df['split'] == 'valid'].copy()
        elif self.split == "test":
            self.df = source_df[source_df['split'] == 'test'].copy()
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # Reset index for clean access
        self.df = self.df.reset_index(drop=True)
    
    def _prepare_metadata(self):
        # Load our species mappings
        species_df = pd.read_excel(self.root_dir / "anura_species_info.xlsx", skiprows=2)
        self.code_to_species = dict(zip(species_df["Code"], species_df["Species"]))

        # Load the main metadata
        if os.path.exists(os.path.join(self.root_dir, "metadata_extra.csv")):
            df = pd.read_csv(self.root_dir / "metadata_extra.csv")
        else:
            df = pd.read_csv(self.root_dir / "metadata.csv")

        # Add the new columns that occur in our mapping and csv to a list
        code_columns = []
        for col in df.columns:
            if col in self.code_to_species:
                code_columns.append(col)
        
        # Replace the code columns to be species names
        for code_col in code_columns:
            df = df.rename(columns={code_col: self.code_to_species[code_col]})

        # Get label columns (for the species)
        label_columns = []
        for col in df.columns[8:]:
            if col in self.code_to_species.values():
                label_columns.append(col)
        
        # Store the labels at the class level
        AnuraDataset._label_columns = label_columns

        # Add the audio_path and other columns if we don't have them
        needs_update = False
        
        if "audio_path" not in df.columns:
            df["audio_path"] = (str(self.root_dir) + "/audio/" + df[self.station_column] + "/" + df[self.audio_column] + "_" + df['min_t'].astype(str) + "_" + df['max_t'].astype(str) + ".wav")
            needs_update = True
        
        if "task" not in df.columns:
            df["task"] = "species-multiple-detection"
            needs_update = True
        
        if "instruction" not in df.columns:
            df["instruction"] = "<Audio><AudioHere></Audio> What are the scientific name(s) for the species in the audio, if any?"
            needs_update = True

        if "output" not in df.columns:
            df["output"] = self._create_output_column(df, AnuraDataset._label_columns)
            needs_update = True

        # Add split column if it doesn't exist (for first-time setup)
        if "split" not in df.columns:
            print("No 'split' column found. Creating predefined splits...")
            df = self._create_predefined_splits(df)
            needs_update = True

        if needs_update:
            self._save_metadata_extra(df)

        # Store the full dataframe at class level for split reference
        AnuraDataset._full_df = df
    
    def _create_predefined_splits(self, df):
        """
        Create a 'split' column with predefined train/valid/test assignments.
        This ensures consistent splits across runs.
        """
        # Create stratification labels (presence/absence of any species)
        stratify_labels = df[AnuraDataset._label_columns].sum(axis=1) > 0
        
        # First split: 80% train+val, 20% test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=0.2,  # 20% for test
            random_state=42, 
            stratify=stratify_labels
        )
        
        # Second split: from the 80%, take 75% for train, 25% for validation
        # This results in: 60% train, 20% validation, 20% test
        train_val_stratify = train_val_df[AnuraDataset._label_columns].sum(axis=1) > 0
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=0.25,  # 25% of train_val = 20% of total
            random_state=42, 
            stratify=train_val_stratify
        )
        
        # Add split column to each dataframe
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'valid'
        test_df['split'] = 'test'
        
        # Combine back
        df_with_splits = pd.concat([train_df, val_df, test_df], axis=0)
        
        print("="*30)
        print("Predefined splits created and added to metadata:")
        print(f"\tTrain: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"\tValid: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"\tTest: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        print("="*30)
        
        return df_with_splits

    @classmethod
    def reset_class_state(cls):
        """Reset class variables - useful for testing different percentage values"""
        cls._train_df = None
        cls._valid_df = None
        cls._test_df = None
        cls._full_df = None
        cls._reduced_df = None
        cls._label_columns = None
        cls._is_prepared = False
        print("Class state reset")

    def _load_predefined_splits(self):
        """Load data based on pre-defined split column"""
        df = AnuraDataset._full_df
        
        # Apply percentage reduction if specified
        if self.percentage is not None:
            # For each split, take a random subset
            split_df = df[df['split'] == self.split].copy()
            
            # Stratify by presence/absence when sampling
            stratify_labels = split_df[AnuraDataset._label_columns].sum(axis=1) > 0
            
            # Calculate sample size
            sample_size = int(len(split_df) * self.percentage)
            
            # Sample while maintaining class distribution
            _, sampled_df = train_test_split(
                split_df,
                test_size=sample_size,
                random_state=42,
                stratify=stratify_labels
            )
            self.df = sampled_df
        else:
            # Use the full split
            self.df = df[df['split'] == self.split].copy()

    def _load_random_splits(self):
        """Legacy method: create random splits on the fly"""
        df = AnuraDataset._full_df
        
        if self.percentage is not None:
            # Create stratification labels for percentage reduction
            current_stratify = df[AnuraDataset._label_columns].sum(axis=1) > 0
            _, df = train_test_split(
                df, 
                test_size=self.percentage, 
                random_state=42, 
                stratify=current_stratify
            )

        # Create stratification labels for the potentially reduced dataframe
        stratify_labels = df[AnuraDataset._label_columns].sum(axis=1) > 0

        # Split the data
        train_val_df, test_df = train_test_split(
            df, 
            test_size=0.1, 
            random_state=42, 
            stratify=stratify_labels
        )
        
        train_val_stratify = train_val_df[AnuraDataset._label_columns].sum(axis=1) > 0
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=0.2, 
            random_state=42, 
            stratify=train_val_stratify
        )

        # Store splits at the class level
        AnuraDataset._train_df = train_df
        AnuraDataset._valid_df = val_df
        AnuraDataset._test_df = test_df

        # Assign the requested split
        if self.split == "train":
            self.df = AnuraDataset._train_df
        elif self.split == "valid":
            self.df = AnuraDataset._valid_df
        elif self.split == "test":
            self.df = AnuraDataset._test_df
    
    def _create_output_column(self, df, label_columns):
        outputs = []
        for idx, row in df[label_columns].iterrows():
            # Get species names where we have a 1 (the species occurs in the given set)
            species_pres = []
            for col in label_columns:
                if row[col] == 1:
                    species_pres.append(col)
            
            if species_pres:
                outputs.append(", ".join(species_pres))
            else:
                outputs.append("None")
        
        return outputs

    def _save_metadata_extra(self, df):
        output_path = self.root_dir / "metadata_extra.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved updated metadata to {output_path}")

    def load_audio(self, audio_path):
        """Load audio for fine-tuning and preprocess it"""
        try:
            # Load audio file
            wav, sr = sf.read(audio_path)

            # Convert to mono if we are in stereo
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)

            # Resample since the audio from the dataset is 22.05 KHz and we need 16KHz for the model
            if sr != self.sample_rate:
                wav_tensor = torch.from_numpy(wav).float()
                resampler = T.Resample(sr, self.sample_rate)
                wav_tensor = resampler(wav_tensor.unsqueeze(0)).squeeze(0)
                wav = wav_tensor.numpy()
            else:
                wav_tensor = torch.from_numpy(wav).float()

            # Pad or truncate
            if len(wav) < self.max_length_samples:
                wav = np.pad(wav, (0, self.max_length_samples - len(wav)))
            else:
                if self.split == "train":
                    # Use random cropping for training
                    start = np.random.randint(0, len(wav) - self.max_length_samples)
                    wav = wav[start:start + self.max_length_samples]
                else:
                    # Center crop for validation or testing
                    start = (len(wav) - self.max_length_samples) // 2
                    wav = wav[start:start + self.max_length_samples]
            
            return torch.from_numpy(wav).float()
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(self.max_length_samples, dtype=torch.float32)
        
    def get_labels(self, row):
        """Extract the species label"""
        labels = []
        for col in self.label_columns:
            if row[col] == 1: # Only add if species is present
                labels.append(col)
        
        if not labels:
            return "None"

        return ", ".join(labels) # Join the labels in a string separated by commas
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Get the audio path
        audio_path = row["audio_path"]

        # Load in the audio
        audio = self.load_audio(audio_path)

        # Get the label(s)
        labels = self.get_labels(row)

        # Extract the relevant features
        return {
            "raw_wav": [audio],
            "text": labels,
            "prompt": self.config.model.prompt_template,
            "task": "species-classification",
            "id": audio_path,
            "index": index
        }
import os
from pathlib import Path

import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T

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

current_dir = Path.cwd()
from NatureLMaudio.NatureLM.dataset import collater

class MacaqueDataset(Dataset):
    _train_df = None
    _valid_df = None
    _test_df = None
    _label_columns = None
    _is_prepared = False

    def __init__(self, config, percentage=None, split="train", root_dir="data/macaques", use_predefined_splits=True, valid_split_ratio=0.2, seed=42):
        self.config = config
        self.percentage = percentage
        self.split = split
        self.root_dir = Path(root_dir)
        self.sample_rate = 16000
        self.max_length_samples = 10 * self.sample_rate
        self.audio_column = "filename"
        self.collater = collater
        self.use_predefined_splits = use_predefined_splits
        self.valid_split_ratio = valid_split_ratio
        self.seed = seed

        if not MacaqueDataset._is_prepared:
            self._prepare_metadata()

        if use_predefined_splits:
            self._load_predefined_splits()
        else:
            self._load_random_splits()

        self.label_columns = MacaqueDataset._label_columns

        print(f"Loaded {self.split} split: {len(self.df)} samples")
        if len(self.label_columns) > 0:
            print(f"Call type: {self.label_columns[0]}")

    def _prepare_metadata(self):
        metadata_path = self.root_dir / "metadata.csv"
        needs_regeneration = True
        
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            # Check if we need to regenerate the splits
            # Regenerate if 'test' split is empty or if split logic needs updating
            split_counts = df['split'].value_counts().to_dict()
            if split_counts.get('test', 0) == 0 or split_counts.get('valid', 0) == 0:
                print("Existing metadata has empty test/valid splits. Regenerating with proper splits...")
                needs_regeneration = True
            else:
                print(f"Loading existing metadata with splits: Train={split_counts.get('train', 0)}, Valid={split_counts.get('valid', 0)}, Test={split_counts.get('test', 0)}")
                needs_regeneration = False
        
        if needs_regeneration or not metadata_path.exists():
            print("Creating/Regenerating metadata with proper train/valid/test splits...")
            audio_files = []
            
            # Handle the case where we only have train and valid folders
            # We'll treat 'valid' as the test set and split 'train' into train/valid
            for folder_name in ['train', 'valid']:
                folder_path = self.root_dir / folder_name
                if folder_path.exists():
                    for audio_file in folder_path.glob("*.wav"):
                        # The 'valid' folder will become our test set
                        split_assignment = 'test' if folder_name == 'valid' else 'train'
                        audio_files.append({
                            'filename': audio_file.stem,
                            'audio_path': str(audio_file),
                            'original_folder': folder_name,
                            'split': split_assignment,  # Will be updated for train split later
                            'call_type': 'coo_call'
                        })

            df = pd.DataFrame(audio_files)
            
            # Now split the 'train' data into actual train and validation sets
            train_mask = df['split'] == 'train'
            train_df = df[train_mask].copy()
            
            if len(train_df) > 0:
                # Create validation split from training data
                train_indices, valid_indices = train_test_split(
                    train_df.index,
                    test_size=self.valid_split_ratio,
                    random_state=self.seed,
                    stratify=None  # All are coo_calls, so stratification isn't needed
                )
                
                # Update split assignments
                df.loc[train_indices, 'split'] = 'train'
                df.loc[valid_indices, 'split'] = 'valid'
            
            df.to_csv(metadata_path, index=False)
            print(f"Created/Updated metadata file at {metadata_path}")
            print(f"Split summary:")
            print(f"  Train: {len(df[df['split'] == 'train'])} samples (from original train folder)")
            print(f"  Valid: {len(df[df['split'] == 'valid'])} samples (from original train folder)")
            print(f"  Test: {len(df[df['split'] == 'test'])} samples (from original valid folder)")
        else:
            df = pd.read_csv(metadata_path)

        label_columns = ['coo_call']
        MacaqueDataset._label_columns = label_columns

        # Ensure label columns exist
        for col in label_columns:
            if col not in df.columns:
                df[col] = (df['call_type'] == col).astype(int)

        needs_update = False

        if "task" not in df.columns:
            df["task"] = "macaque-call-classification"
            needs_update = True

        if "instruction" not in df.columns:
            df["instruction"] = "<Audio><AudioHere></Audio> What type of macaque call is present in the audio, if any?"
            needs_update = True

        if "output" not in df.columns:
            df["output"] = self._create_output_column(df, MacaqueDataset._label_columns)
            needs_update = True

        if needs_update:
            self._save_metadata_extra(df)
            # Also update the main metadata file
            df.to_csv(metadata_path, index=False)

        MacaqueDataset._full_df = df

    def _create_predefined_splits(self, df):
        """This method is no longer needed as splits are created in _prepare_metadata"""
        return df

    def _load_predefined_splits(self):
        df = MacaqueDataset._full_df

        if self.percentage is not None:
            split_df = df[df['split'] == self.split].copy()
            
            # Since all samples are coo_calls, we don't need stratification
            sample_size = int(len(split_df) * self.percentage)
            
            if sample_size > 0 and sample_size < len(split_df):
                _, sampled_df = train_test_split(
                    split_df,
                    test_size=sample_size,
                    random_state=self.seed
                )
                self.df = sampled_df
            else:
                self.df = split_df
        else:
            self.df = df[df['split'] == self.split].copy()

    def _filter_short_audio(self, df, min_duration=0.5):
        """Filter out audio files shorter than minimum duration"""
        original_count = len(df)
        
        def get_duration(audio_path):
            try:
                info = sf.info(audio_path)
                return info.duration
            except:
                return 0
        
        df['duration'] = df['audio_path'].apply(get_duration)
        df = df[df['duration'] >= min_duration].copy()
        df = df.drop(columns=['duration'])
        
        filtered_count = len(df)
        if original_count != filtered_count:
            print(f"Filtered out {original_count - filtered_count} audio files shorter than {min_duration}s")
        
        return df

    def _load_random_splits(self):
        df = MacaqueDataset._full_df

        if self.percentage is not None:
            _, df = train_test_split(
                df,
                test_size=self.percentage,
                random_state=self.seed
            )

        # Split into train (70%), validation (15%), test (15%)
        train_val_df, test_df = train_test_split(
            df,
            test_size=0.15,
            random_state=self.seed
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.valid_split_ratio,
            random_state=self.seed
        )

        MacaqueDataset._train_df = train_df
        MacaqueDataset._valid_df = val_df
        MacaqueDataset._test_df = test_df

        if self.split == "train":
            self.df = MacaqueDataset._train_df
        elif self.split == "valid":
            self.df = MacaqueDataset._valid_df
        elif self.split == "test":
            self.df = MacaqueDataset._test_df

    def _create_output_column(self, df, label_columns):
        outputs = []
        for idx, row in df[label_columns].iterrows():
            call_types_pres = []
            for col in label_columns:
                if row[col] == 1:
                    call_types_pres.append(col)

            if call_types_pres:
                outputs.append(", ".join(call_types_pres))
            else:
                outputs.append("None")

        return outputs

    def _save_metadata_extra(self, df):
        output_path = self.root_dir / "metadata_extra.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved updated metadata to {output_path}")

    def load_audio(self, audio_path):
        try:
            wav, sr = sf.read(audio_path)

            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)

            if sr != self.sample_rate:
                wav_tensor = torch.from_numpy(wav).float()
                resampler = T.Resample(sr, self.sample_rate)
                wav_tensor = resampler(wav_tensor.unsqueeze(0)).squeeze(0)
                wav = wav_tensor.numpy()
            else:
                wav_tensor = torch.from_numpy(wav).float()

            # Ensure minimum length of 0.5 seconds (8000 samples at 16kHz)
            min_length_samples = int(0.5 * self.sample_rate)  # 8000 samples
            
            if len(wav) < min_length_samples:
                # Pad to minimum length
                pad_length = min_length_samples - len(wav)
                wav = np.pad(wav, (0, pad_length), mode='constant', constant_values=0)
            elif len(wav) < self.max_length_samples:
                # Pad to max length if shorter
                wav = np.pad(wav, (0, self.max_length_samples - len(wav)))
            else:
                # Random crop for training, center crop for validation/test
                if self.split == "train":
                    start = np.random.randint(0, len(wav) - self.max_length_samples)
                    wav = wav[start:start + self.max_length_samples]
                else:
                    start = (len(wav) - self.max_length_samples) // 2
                    wav = wav[start:start + self.max_length_samples]

            return torch.from_numpy(wav).float()
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(self.max_length_samples, dtype=torch.float32)

    def get_labels(self, row):
        labels = []
        for col in self.label_columns:
            if row[col] == 1:
                labels.append(col)

        if not labels:
            return "None"

        return ", ".join(labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        audio_path = row["audio_path"]

        audio = self.load_audio(audio_path)

        labels = self.get_labels(row)

        return {
            "raw_wav": [audio],
            "text": labels,
            "prompt": self.config.model.prompt_template,
            "task": "macaque-call-classification",
            "id": audio_path,
            "index": index
        }
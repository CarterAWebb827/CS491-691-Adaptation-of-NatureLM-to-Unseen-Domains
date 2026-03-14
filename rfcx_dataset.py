import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import soundfile as sf
import torchaudio.transforms as T

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
    
    from NatureLM.dataset import collater
else:
    from NatureLMaudio.NatureLM.dataset import collater

class RFCFrogDataset(Dataset):
    """Dataset class for RFCx frog species (species IDs 0-10)"""
    
    # Species mapping: id -> (scientific_name, code) for species 0-10 only
    SPECIES_MAPPING = {
        0: ("Eleutherodactylus unicolor", "ELUN"),
        1: ("Eleutherodactylus brittoni", "ELBR"),
        2: ("Eleutherodactylus wightmanae", "ELWI"),
        3: ("Eleutherodactylus coqui", "ELCO"),
        4: ("Eleutherodactylus hedricki", "ELHE"),
        5: ("Eleutherodactylus gryllus", "ELGR"),
        6: ("Eleutherodactylus richmondi", "ELRI"),
        7: ("Eleutherodactylus portoricensis", "ELPO"),
        8: ("Eleutherodactylus locustus", "ELLO"),
        9: ("Eleutherodactylus antillensis", "ELAN"),
        10: ("Leptodactylus albilabris", "LEAL")
    }
    
    # Create a list of species IDs we care about
    FROG_SPECIES_IDS = list(range(11))  # 0 through 10
    
    def __init__(self, config, split="train", root_dir="data/rfcx", use_fp=False):
        """
        Args:
            config: Configuration object
            split: "train", "test", or "train_all" (train_tp + train_fp)
            root_dir: Root directory containing train/test folders and metadata
            use_fp: If True, include false positives in training (only relevant for split="train")
        """
        self.config = config
        self.split = split
        self.root_dir = Path(root_dir)
        self.sample_rate = 16000
        self.max_length_samples = 10 * self.sample_rate
        self.audio_column = "recording_id"
        self.collater = collater
        self.use_fp = use_fp
        
        # Load the appropriate data based on split
        if split == "train":
            self.df = self._load_train_data()
        elif split == "test":
            self.df = self._load_test_data()
        elif split == "train_all":
            # Load both TP and FP for training
            self.df = self._load_train_data(include_fp=True)
        else:
            raise ValueError(f"Split must be 'train', 'test', or 'train_all', got {split}")
        
        # Create label columns (scientific names) for species 0-10
        self.label_columns = [self.SPECIES_MAPPING[i][0] for i in range(11)]
        
        # Create a mapping from species_id to scientific name
        self.id_to_species = {i: self.SPECIES_MAPPING[i][0] for i in range(11)}
        
        print(f"Loaded {self.split} split: {len(self.df)} samples")
        print(f"Number of frog species: {len(self.label_columns)}")
        print(f"Species: {', '.join([f'{k}: {v[0]}' for k, v in self.SPECIES_MAPPING.items()])}")
    
    def _load_train_data(self, include_fp=False):
        """Load training data (TP and optionally FP)"""
        
        # Load true positives
        tp_path = self.root_dir / "train_tp.csv"
        if not tp_path.exists():
            raise FileNotFoundError(f"Training TP file not found: {tp_path}")
        
        tp_df = pd.read_csv(tp_path)
        tp_df['label_type'] = 'tp'  # Mark as true positive
        
        # Filter for frog species only (0-10)
        tp_df = tp_df[tp_df['species_id'].isin(self.FROG_SPECIES_IDS)].copy()
        
        dataframes = [tp_df]
        
        # Load false positives if requested
        if include_fp:
            fp_path = self.root_dir / "train_fp.csv"
            if fp_path.exists():
                fp_df = pd.read_csv(fp_path)
                fp_df['label_type'] = 'fp'  # Mark as false positive
                # Filter for frog species only
                fp_df = fp_df[fp_df['species_id'].isin(self.FROG_SPECIES_IDS)].copy()
                dataframes.append(fp_df)
                print(f"Added {len(fp_df)} false positive samples")
            else:
                print(f"Warning: FP file not found at {fp_path}")
        
        # Combine dataframes
        df = pd.concat(dataframes, ignore_index=True)
        
        # Create audio path
        df['audio_path'] = df[self.audio_column].apply(
            lambda x: str(self.root_dir / "train" / f"{x}.flac")
        )
        
        # Add task and instruction columns
        df['task'] = "species-multiple-detection"
        df['instruction'] = "<Audio><AudioHere></Audio> What are the scientific name(s) for the species in the audio, if any?"
        
        # For FP samples, output should be "None" (no species present)
        df['output'] = df.apply(lambda row: "None" if row['label_type'] == 'fp' else self.id_to_species[row['species_id']], axis=1)
        
        return df
    
    def _load_test_data(self):
        """Load test data (just the recording IDs from sample submission)"""
        
        # Load sample submission to get test recording IDs
        submission_path = self.root_dir / "sample_submission.csv"
        if not submission_path.exists():
            raise FileNotFoundError(f"Sample submission file not found: {submission_path}")
        
        submission_df = pd.read_csv(submission_path)
        
        # Create a dataframe with just recording_ids
        df = pd.DataFrame({'recording_id': submission_df['recording_id'].unique()})
        
        # Create audio path (test audio files are in test folder)
        df['audio_path'] = df['recording_id'].apply(
            lambda x: str(self.root_dir / "test" / f"{x}.flac")
        )
        
        # Add task and instruction columns
        df['task'] = "species-multiple-detection"
        df['instruction'] = "<Audio><AudioHere></Audio> What are the scientific name(s) for the species in the audio, if any?"
        
        # For test set, output is unknown (we'll use this for inference only)
        df['output'] = "unknown"
        
        return df
    
    def load_audio(self, audio_path):
        """Load audio for fine-tuning and preprocess it"""
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                return torch.zeros(self.max_length_samples, dtype=torch.float32)
            
            # Load audio file
            wav, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sr != self.sample_rate:
                wav_tensor = torch.from_numpy(wav).float()
                resampler = T.Resample(sr, self.sample_rate)
                wav_tensor = resampler(wav_tensor.unsqueeze(0)).squeeze(0)
                wav = wav_tensor.numpy()
            
            # Pad or truncate to max_length_samples
            if len(wav) < self.max_length_samples:
                wav = np.pad(wav, (0, self.max_length_samples - len(wav)))
            else:
                if self.split == "train":
                    # Random crop for training
                    start = np.random.randint(0, len(wav) - self.max_length_samples)
                    wav = wav[start:start + self.max_length_samples]
                else:
                    # Center crop for validation/testing
                    start = (len(wav) - self.max_length_samples) // 2
                    wav = wav[start:start + self.max_length_samples]
            
            return torch.from_numpy(wav).float()
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(self.max_length_samples, dtype=torch.float32)
    
    def get_labels(self, row):
        """Extract the species label(s)"""
        if 'label_type' in row and row['label_type'] == 'fp':
            return "None"
        
        if 'species_id' in row and pd.notna(row['species_id']):
            species_id = int(row['species_id'])
            if species_id in self.id_to_species:
                return self.id_to_species[species_id]
        
        return "None"
    
    def get_binary_labels(self, row):
        """Get binary vector of species presence (useful for multi-label classification)"""
        labels = torch.zeros(len(self.label_columns))
        if 'species_id' in row and pd.notna(row['species_id']):
            species_id = int(row['species_id'])
            if 0 <= species_id <= 10:
                labels[species_id] = 1
        return labels
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        # Get audio path
        audio_path = row["audio_path"]
        
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Get label(s)
        labels = self.get_labels(row)
        
        # Return in format expected by NatureLM
        return {
            "raw_wav": [audio],
            "text": labels,
            "prompt": self.config.model.prompt_template,
            "task": "species-classification",
            "id": audio_path,
            "index": index,
            "recording_id": row['recording_id'],
            "binary_labels": self.get_binary_labels(row)  # Optional: for multi-label loss
        }


class RFCTestDataset(Dataset):
    """
    Special dataset for test submission that returns recording_id
    along with audio for prediction
    """
    def __init__(self, config, root_dir="data/rfcx"):
        self.config = config
        self.root_dir = Path(root_dir)
        self.sample_rate = 16000
        self.max_length_samples = 10 * self.sample_rate
        
        # Load sample submission to get recording IDs
        submission_path = self.root_dir / "sample_submission.csv"
        self.submission_df = pd.read_csv(submission_path)
        
        # Get unique recording IDs
        self.recording_ids = self.submission_df['recording_id'].unique()
        
        print(f"Loaded {len(self.recording_ids)} test samples")
    
    def load_audio(self, audio_path):
        """Load audio for inference"""
        try:
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}")
                return torch.zeros(self.max_length_samples, dtype=torch.float32)
            
            wav, sr = sf.read(audio_path)
            
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            
            if sr != self.sample_rate:
                wav_tensor = torch.from_numpy(wav).float()
                resampler = T.Resample(sr, self.sample_rate)
                wav_tensor = resampler(wav_tensor.unsqueeze(0)).squeeze(0)
                wav = wav_tensor.numpy()
            
            if len(wav) < self.max_length_samples:
                wav = np.pad(wav, (0, self.max_length_samples - len(wav)))
            else:
                # Center crop for test
                start = (len(wav) - self.max_length_samples) // 2
                wav = wav[start:start + self.max_length_samples]
            
            return torch.from_numpy(wav).float()
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(self.max_length_samples, dtype=torch.float32)
    
    def __len__(self):
        return len(self.recording_ids)
    
    def __getitem__(self, index):
        recording_id = self.recording_ids[index]
        audio_path = self.root_dir / "test" / f"{recording_id}.flac"
        
        audio = self.load_audio(audio_path)
        
        return {
            "raw_wav": [audio],
            "recording_id": recording_id,
            "index": index,
            "audio_path": str(audio_path)
        }
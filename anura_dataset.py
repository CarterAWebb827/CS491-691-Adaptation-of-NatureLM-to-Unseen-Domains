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

current_dir = Path.cwd()
anura_dir = Path(os.path.join(current_dir, "data/AnuraSet"))

class AnuraDataset(Dataset):
    # Class variables to store splits across instances of the AnuraDataset class
    _train_df = None
    _valid_df = None
    _test_df = None
    _label_columns = None
    _is_prepared = False

    def __init__(self, config, percentage=None, split="train", root_dir="data/AnuraSet"):
        self.config = config
        self.percentage = percentage
        self.split = split
        self.root_dir = Path(root_dir)
        self.sample_rate = 16000
        self.max_length_samples = 10 * self.sample_rate
        self.audio_column = "fname"
        self.station_column = "site"
        self.collater = collater

        # Prepare the metadata
        if not AnuraDataset._is_prepared:
            self._prepare_metadata()

        # Assign the appropriate splits
        if self.split == "train":
            self.df = AnuraDataset._train_df
        elif self.split == "valid":
            self.df = AnuraDataset._valid_df
        elif self.split == "test":
            self.df = AnuraDataset._test_df

        self.label_columns = AnuraDataset._label_columns

        print(f"Loaded {self.split} split: {len(self.df)} samples")
        print(f"Number of species: {len(self.label_columns)}")
    
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

        # Add the audio_path and/or task column if we dont have it
        if "audio_path" not in df.columns or "task" not in df.columns or "instruction" not in df.columns:
            if "audio_path" not in df.columns:
                df["audio_path"] = (str(self.root_dir) + "/audio/" + df[self.station_column] + "/" + df[self.audio_column] + "_" + df['min_t'].astype(str) + "_" + df['max_t'].astype(str) + ".wav")
            
            if "task" not in df.columns:
                # df["task"] = "species-sci-options-classification"
                df["task"] = "species-multiple-detection"
            
            if "instruction" not in df.columns:
                df["instruction"] = "<Audio><AudioHere></Audio> What are the scientific name(s) for the species in the audio, if any?"

            if "output" not in df.columns:
                df["output"] = self._create_output_column(df, AnuraDataset._label_columns)

            self._save_metadata_extra(df)

        if self.percentage is not None:
            # Create startification labels (presence/absence)
            # Stratification means that our train/valid/test splits will contain the same proportion of each class label as the original dataset (no skew)
            current_stratify = df[AnuraDataset._label_columns].sum(axis=1) > 0
            _, df= train_test_split(df, test_size=self.percentage, random_state=42, stratify=current_stratify)

        # Create stratification labels for the potentially reduced dataframe to avoid size mismatch error
        stratify_labels = df[AnuraDataset._label_columns].sum(axis=1) > 0

        # Split the data
        # First, split between training/validation and test data
        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=stratify_labels)
        
        # Next, split between the training and validation data
        train_val_stratify = train_val_df[AnuraDataset._label_columns].sum(axis=1) > 0
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_stratify)

        # Store splits at the class level
        AnuraDataset._train_df = train_df
        AnuraDataset._valid_df = val_df
        AnuraDataset._test_df = test_df
        AnuraDataset._is_prepared = True

        print("="*30)
        print("Dataset splits created:")
        print(f"\tTrain: {len(train_df)} samples")
        print(f"\tValid: {len(val_df)} samples")
        print(f"\tTest: {len(test_df)} samples")
        print("="*30)
    
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

    def load_audio(self, audio_path):
        """Load audio for fine-tuning and preprocess it"""
        try:
            # Load audo file
            wav, sr = sf.read(audio_path)

            # Convert to mono if we are in stereo
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)

            # Resample since the audio from the dataset is 22.05 KHz and we need 16KHz for the model
            if sr != self.sample_rate:
                wav_tensor = torch.from_numpy(wav).float()
                resampler = T.Resample(sr, self.sample_rate)
                sr = self.sample_rate
                wav_tensor = resampler(wav_tensor.unsqueeze(0)).squeeze(0)
                wav = wav_tensor.numpy()
            else:
                wav_tensor = torch.from_numpy(wav).float()

            # Pad or truncate
            if len(wav) < self.max_length_samples:
                wav = np.pad(wav, (0, self.max_length_samples - len(wav),))
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
            return torch.zeros(self.max_length_samples, dtype=np.float32)
        
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
        # audio_filename = row[self.audio_column]
        # station = row[self.station_column]
        # min_t = row["min_t"]
        # max_t = row["max_t"]
        # audio_path = Path(f"{self.root_dir}/audio/{station}/{audio_filename}_{min_t}_{max_t}.wav")
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
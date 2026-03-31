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
fasd13_dir = Path(os.path.join(current_dir, "drive/MyDrive/FASD13"))

class FASD13Dataset(Dataset):
    """Dataset class for FASD13 (species IDs 0-12)"""

    COMMON_NAME_MAPPING = {
        0:  ("AnuraSet", "AS"),
        1:  ("Carrion Crow", "CC"), 
        2:  ("Gunshot", "GS"), 
        3:  ("Hawaiian bird", "HA"), 
        4:  ("Hainan Gibbon", "HG"), 
        5:  ("Humpback Whale", "HW"), 
        6:  ("Jumping Spider", "JS"),
        7:  ("Katydid", "KD"), 
        8:  ("Marmoset", "MS"), 
        9:  ("Powdermill", "PM"), 
        10: ("Ruffed Grouse", "RG"), 
        11: ("Rana Sierrae", "RS"), 
        12: ("Right Whale", "RW"),
    }

    COMMON_NAME_IDS = list(range(13))

    # Class variables to store splits across instances of the FASD13Dataset class
    _train_df = None
    _valid_df = None
    _test_df = None
    _label_columns = None
    _is_prepared = False

    def __init__(self, config, percentage=None, split="train", root_dir="drive/MyDrive/FASD13"):
        self.config = config
        self.percentage = percentage
        self.split = split
        self.root_dir = Path(getattr(config, "data_dir", root_dir))
        self.sample_rate = 16000
        self.max_length_samples = 10 * self.sample_rate
        self.audio_column = "fname"
        self.station_column = "site"
        self.collater = collater

        #prepare metadata
        if not FASD13Dataset._is_prepared:
            self._prepare_metadata()

        # Assign the appropriate splits
        if self.split == "train":
            self.df = FASD13Dataset._train_df
        elif self.split == "valid":
            self.df = FASD13Dataset._valid_df
        elif self.split == "test":
            self.df = FASD13Dataset._test_df

        self.label_columns = FASD13Dataset._label_columns

        print(f"Loaded {self.split} split: {len(self.df)} samples")
        print(f"Number of species: {len(self.label_columns)}")
    
    def _prepare_metadata(self):
        print("Root directory:", self.root_dir)
        for _, (species_name, code) in self.COMMON_NAME_MAPPING.items():
            dataset_dir = self.root_dir / code
            print(f"{species_name} dir exists? {dataset_dir.exists()}")
            print(f"CSV files: {list(dataset_dir.glob('*.csv'))}")
        #directories are /content/drive/FASD13/*
            #* is AS, CC, GS, HA, HG, HW, JS, KD, MS, PM, RG, RS, RW
        #files are .csv and .wav with matching names
            #columns:
                #Starttime - in seconds
                #Endtime - in seconds
                #Q - POS, 
                #Audiofilename - name of matched audio file

        #check datadir / folder name for all .csv files
        #parse the .csv files to index beginning, end, and matched .wav filenames
        #split off the .wav at the beginning and end
        #store that wav and the name of the dataset it comes from in a data structure

        rows = []
        # Iterate over each dataset directory (AS, CC, GS, etc.)
        for _, (species_name, code) in self.COMMON_NAME_MAPPING.items():
            dataset_dir = self.root_dir / code
            if not dataset_dir.exists():
                print(f"Warning: dataset directory not found: {dataset_dir}")
                continue

            csv_files = list(dataset_dir.glob("*.csv"))

            for csv_path in csv_files:
                df = pd.read_csv(csv_path)

                # Expect columns: Starttime, Endtime, Q, Audiofilename
                for _, r in df.iterrows():

                    start = r["Starttime"]
                    end = r["Endtime"]
                    audio_file = r["Audiofilename"]

                    wav_path = dataset_dir / audio_file

                    # Create one-hot label row
                    label_dict = {}
                    for _, (sp_name, _) in self.COMMON_NAME_MAPPING.items():
                        label_dict[sp_name] = 1 if sp_name == species_name else 0

                    #common name is what is given by this dataset
                    rows.append({
                        "audio_path": str(wav_path),
                        "start_time": start,
                        "end_time": end,
                        "task": "species-multiple-detection",
                        "instruction": "<Audio><AudioHere></Audio> What are the common name(s) for the species in the audio, if any?",
                        "output": species_name,
                        **label_dict
                    })

        if len(rows) == 0:
            raise ValueError(f"No data found in {self.root_dir}. Check your CSV paths and folder structure.")
        df = pd.DataFrame(rows)

        # Determine label columns
        label_columns = [v[0] for v in self.COMMON_NAME_MAPPING.values()]
        FASD13Dataset._label_columns = label_columns

        # Stratification labels (presence/absence)
        stratify_labels = df[label_columns].sum(axis=1) > 0

        # Train/test split
        train_val_df, test_df = train_test_split(
            df, test_size=0.1, random_state=42, stratify=stratify_labels
        )

        train_val_stratify = train_val_df[label_columns].sum(axis=1) > 0

        train_df, val_df = train_test_split(
            train_val_df, test_size=0.2, random_state=42, stratify=train_val_stratify
        )

        FASD13Dataset._train_df = train_df
        FASD13Dataset._valid_df = val_df
        FASD13Dataset._test_df = test_df
        FASD13Dataset._is_prepared = True

        print("=" * 30)
        print("Dataset splits created:")
        print(f"\tTrain: {len(train_df)} samples")
        print(f"\tValid: {len(val_df)} samples")
        print(f"\tTest: {len(test_df)} samples")
        print("=" * 30)

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

    def load_audio(self, audio_path):
        """Load audio for fine-tuning and preprocess it"""
        try:
            # Load audo file
            wav, sr = sf.read(audio_path)

            # Convert to mono if we are in stereo
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)

            # Resample since we need 16KHz for the model
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
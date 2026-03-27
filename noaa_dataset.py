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
noaa_dir = Path(os.path.join(current_dir, "drive/MyDrive/RightWhaleData"))

class RightWhaleDataset(Dataset):

    # Class variables to store splits across instances of the FASD13Dataset class
    _train_df = None
    _valid_df = None
    _test_df = None
    _label_columns = None
    _is_prepared = False

    def __init__(self):
        pass

    def _prepare_metadata(self):
        pass

    def _create_output_column(self, df, label_columns):
        pass

    def load_audio(self, audio_path):
        pass

    def get_labels(self, row):
        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
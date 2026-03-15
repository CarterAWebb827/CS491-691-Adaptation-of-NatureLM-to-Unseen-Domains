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
    data_dir = current_dir / "drive" / "My Drive" / "FASD13"
    
    from NatureLM.dataset import collater
else:
    from NatureLMaudio.NatureLM.dataset import collater

class FASD13Dataset(Dataset):
    """Dataset class for FASD13 (species IDs 0-12)"""

    SPECIES_MAPPING = {
        0:  ("AnuraSet", "AS")
        1:  ("Carrion Crow", "CC"), 
        2:  ("Gunshot", "GS"), 
        3:  ("Hawaiian bird", "HA"), 
        4:  ("Hainan Gibbon", "HG"), 
        5:  ("Humpback Whale", "HW"), 
        6:  ("Jumping Spider", "JS"),
        7:  ("Katydid", "KD"), 
        8:  ("Marmoset", "MS"), 
        9:  ("Powdermill", "PM"), 
        10:  ("Ruffed Grouse", "RG"), 
        11: ("Rana Sierrae", "RS"), 
        12: ("Right Whale", "RW")
    }

    SPECIES_IDS = list(range(12))

    def __init__(self):
        #directories are /content/drive/FASD13/*
            #* is AS, CC, GS, HA, HG, HW, JS, KD, MS, PM, RG, RS, RW
        #files are .csv and .wav with matching names
            #columns:
                #Starttime - in seconds
                #Endtime - in seconds
                #Audiofilename - name of matched audio file

        #check datadir / folder name for all .csv files
        #parse the .csv files to index beginning, end, and matched .wav filenames
        #split off the .wav at the beginning and end
        #store that wav and the name of the dataset it comes from in a data structure


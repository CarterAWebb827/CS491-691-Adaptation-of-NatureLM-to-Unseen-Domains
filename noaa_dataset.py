import os
import re
import sys
import wave
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    current_dir = Path.cwd()
    print(f{str(current_dir)})
    naturelm_dir = Path("content/drive/MyDrive/NatureLMaudio")
    print(f{str(naturelm_dir)})
    if str(naturelm_dir) not in sys.path:
        sys.path.insert(0, str(naturelm_dir))
        print(f"Added {naturelm_dir} to Python path")

    from NatureLM.dataset import collater
else:
    from NatureLMaudio.NatureLM.dataset import collater

current_dir = Path.cwd()
noaa_dir = Path(os.path.join("content/drive/MyDrive/RightWhaleData"))


class RightWhaleDataset(Dataset):
    SPECIES_NAME = "Right Whale"
    RIWH_CODE = "RIWH"
    LOG_SUFFIX = "_upcall-detection-log.csv"

    _train_df = None
    _valid_df = None
    _test_df = None
    _label_columns = None
    _is_prepared = False

    def __init__(self, config, percentage=None, split="train", root_dir="drive/MyDrive/RightWhaleData"):
        self.config = config
        self.percentage = percentage
        self.split = split
        self.root_dir = Path(getattr(config, "data_dir", root_dir))
        self.sample_rate = 16000
        self.max_length_samples = 10 * self.sample_rate
        self.clip_duration_seconds = 10
        self.collater = collater

        if not RightWhaleDataset._is_prepared:
            self._prepare_metadata()

        if self.split == "train":
            self.df = RightWhaleDataset._train_df
        elif self.split == "valid":
            self.df = RightWhaleDataset._valid_df
        elif self.split == "test":
            self.df = RightWhaleDataset._test_df
        else:
            raise ValueError(f"Split must be 'train', 'valid', or 'test', got {self.split}")

        self.label_columns = RightWhaleDataset._label_columns

        print(f"Loaded {self.split} split: {len(self.df)} samples")
        print(f"Number of species: {len(self.label_columns)}")

    def _prepare_metadata(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"RightWhaleData directory not found: {self.root_dir}. "
                "Mount Google Drive in Colab before creating the dataset."
            )

        metadata_cache = self.root_dir / "metadata_extra.csv"
        if metadata_cache.exists():
            df = pd.read_csv(metadata_cache)
            for column in ["detection_start_time", "detection_end_time", "chunk_start_time", "chunk_end_time"]:
                if column in df.columns:
                    df[column] = pd.to_datetime(df[column])
        else:
            df = self._build_metadata_dataframe()
            self._save_metadata_extra(df)

        RightWhaleDataset._label_columns = [self.SPECIES_NAME]

        if self.percentage is not None:
            df = self._sample_dataframe(df, self.percentage)

        train_df, val_df, test_df = self._split_dataframe(df)

        RightWhaleDataset._train_df = train_df
        RightWhaleDataset._valid_df = val_df
        RightWhaleDataset._test_df = test_df
        RightWhaleDataset._is_prepared = True

        print("=" * 30)
        print("Dataset splits created:")
        print(f"\tTrain: {len(train_df)} samples")
        print(f"\tValid: {len(val_df)} samples")
        print(f"\tTest: {len(test_df)} samples")
        print("=" * 30)

    def _build_metadata_dataframe(self):
        rows = []

        for dataset_dir in sorted(self.root_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            audio_dir = dataset_dir / "ancillary" / "source-audio"
            if not audio_dir.exists():
                continue

            wav_paths = sorted(audio_dir.glob("*.wav"))
            if not wav_paths:
                print(f"Skipping {dataset_dir.name}: no source audio files found")
                continue

            wav_index = self._build_wav_index(wav_paths)
            log_path = self._select_log_file(dataset_dir / "data")
            if log_path is None:
                print(f"Skipping {dataset_dir.name}: no canonical upcall log found")
                continue

            detections_df = pd.read_csv(log_path)
            detections_df = self._normalize_detection_log(detections_df)

            for _, row in detections_df.iterrows():
                record = self._build_record_from_detection(row, dataset_dir, log_path, wav_index)
                if record is not None:
                    rows.append(record)

        if not rows:
            raise ValueError(f"No usable NOAA rows found in {self.root_dir}")

        df = pd.DataFrame(rows)
        df[self.SPECIES_NAME] = 1
        df["task"] = "species-multiple-detection"
        df["instruction"] = "<Audio><AudioHere></Audio> What are the common name(s) for the species in the audio, if any?"
        df["output"] = self._create_output_column(df, [self.SPECIES_NAME])
        df["dataset_name"] = "noaa-right-whale"

        ordered_columns = [
            "audio_path",
            "clip_start_seconds",
            "clip_end_seconds",
            "detection_start_time",
            "detection_end_time",
            "chunk_start_time",
            "chunk_end_time",
            "task",
            "instruction",
            "output",
            self.SPECIES_NAME,
            "dataset_name",
            "source_csv",
            "dataset_dir",
            "selection",
            "species_code",
            "detection_confidence",
            "channel",
            "low_freq_hz",
            "high_freq_hz",
        ]
        return df[ordered_columns]

    def _select_log_file(self, data_dir):
        log_files = sorted(
            path for path in data_dir.glob(f"*{self.LOG_SUFFIX}")
            if "depricated" not in path.name.lower()
        )
        if not log_files:
            return None
        return log_files[0]

    def _normalize_detection_log(self, df):
        required_columns = [
            "Selection",
            "Channel",
            "Start_DateTime_ISO8601",
            "End_DateTime_ISO8601",
            "Low.Freq..Hz.",
            "High.Freq..Hz.",
            "Species",
            "Detection_Confidence",
        ]

        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing expected NOAA columns: {missing}")

        df = df[required_columns].copy()
        df = df[df["Species"] == self.RIWH_CODE].copy()
        df["Start_DateTime_ISO8601"] = pd.to_datetime(df["Start_DateTime_ISO8601"], utc=True)
        df["End_DateTime_ISO8601"] = pd.to_datetime(df["End_DateTime_ISO8601"], utc=True)

        # Some logs contain paired Waveform/Spectrogram rows for the same detection.
        df = df.drop_duplicates(
            subset=[
                "Selection",
                "Channel",
                "Start_DateTime_ISO8601",
                "End_DateTime_ISO8601",
                "Low.Freq..Hz.",
                "High.Freq..Hz.",
                "Species",
                "Detection_Confidence",
            ]
        )

        return df.reset_index(drop=True)

    def _build_wav_index(self, wav_paths):
        wav_index = []
        for wav_path in wav_paths:
            start_time = self._parse_wav_start_time(wav_path.name)
            duration_seconds = self._get_wav_duration_seconds(wav_path)
            end_time = start_time + timedelta(seconds=duration_seconds)
            wav_index.append(
                {
                    "path": wav_path,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
        return wav_index

    def _parse_wav_start_time(self, wav_name):
        match = re.search(r"(20\d{6})_(\d{6})", wav_name)
        if match:
            return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")

        match = re.search(r"(\d{6})-(\d{6})", wav_name)
        if match:
            return datetime.strptime("".join(match.groups()), "%y%m%d%H%M%S")

        raise ValueError(f"Could not parse timestamp from NOAA wav filename: {wav_name}")

    def _get_wav_duration_seconds(self, wav_path):
        with wave.open(str(wav_path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())

    def _build_record_from_detection(self, row, dataset_dir, log_path, wav_index):
        detection_start = row["Start_DateTime_ISO8601"].tz_localize(None)
        detection_end = row["End_DateTime_ISO8601"].tz_localize(None)

        wav_entry = self._find_wav_for_detection(detection_start, detection_end, wav_index)
        if wav_entry is None:
            print(
                f"Skipping detection in {log_path.name}: "
                f"no source wav covers {detection_start.isoformat()} -> {detection_end.isoformat()}"
            )
            return None

        clip_start_seconds = max(0.0, (detection_start - wav_entry["start_time"]).total_seconds())
        clip_end_seconds = max(clip_start_seconds, (detection_end - wav_entry["start_time"]).total_seconds())

        return {
            "audio_path": str(wav_entry["path"]),
            "clip_start_seconds": float(clip_start_seconds),
            "clip_end_seconds": float(clip_end_seconds),
            "detection_start_time": detection_start,
            "detection_end_time": detection_end,
            "chunk_start_time": wav_entry["start_time"],
            "chunk_end_time": wav_entry["end_time"],
            "source_csv": str(log_path),
            "dataset_dir": str(dataset_dir),
            "selection": int(row["Selection"]),
            "species_code": row["Species"],
            "detection_confidence": row["Detection_Confidence"],
            "channel": int(row["Channel"]),
            "low_freq_hz": float(row["Low.Freq..Hz."]),
            "high_freq_hz": float(row["High.Freq..Hz."]),
        }

    def _find_wav_for_detection(self, detection_start, detection_end, wav_index):
        for entry in wav_index:
            if entry["start_time"] <= detection_end <= entry["end_time"]:
                return entry

        return None

    def _sample_dataframe(self, df, percentage):
        if percentage <= 0 or percentage >= 1:
            return df.reset_index(drop=True)

        _, sampled_df = train_test_split(df, test_size=percentage, random_state=42)
        return sampled_df.reset_index(drop=True)

    def _split_dataframe(self, df):
        train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def _create_output_column(self, df, label_columns):
        outputs = []
        for _, row in df[label_columns].iterrows():
            labels = [column for column in label_columns if row[column] == 1]
            outputs.append(", ".join(labels) if labels else "None")
        return outputs

    def _save_metadata_extra(self, df):
        output_df = df.copy()
        for column in ["detection_start_time", "detection_end_time", "chunk_start_time", "chunk_end_time"]:
            output_df[column] = output_df[column].astype(str)
        output_df.to_csv(self.root_dir / "metadata_extra.csv", index=False)

    def load_audio(self, audio_path, start_time=None, end_time=None):
        try:
            if not os.path.exists(audio_path):
                print(f"Warning: audio file not found: {audio_path}")
                return torch.zeros(self.max_length_samples, dtype=torch.float32)

            info = sf.info(audio_path)
            frame_start = 0 if start_time is None else max(0, int(float(start_time) * info.samplerate))
            frame_stop = info.frames if end_time is None else min(info.frames, int(float(end_time) * info.samplerate))

            if frame_stop <= frame_start:
                frame_stop = min(info.frames, frame_start + int(self.clip_duration_seconds * info.samplerate))

            wav, sr = sf.read(audio_path, start=frame_start, stop=frame_stop)

            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            if sr != self.sample_rate:
                wav_tensor = torch.from_numpy(wav).float()
                resampler = T.Resample(sr, self.sample_rate)
                wav_tensor = resampler(wav_tensor.unsqueeze(0)).squeeze(0)
                wav = wav_tensor.numpy()

            if len(wav) < self.max_length_samples:
                wav = np.pad(wav, (0, self.max_length_samples - len(wav)))
            else:
                if self.split == "train":
                    max_offset = len(wav) - self.max_length_samples
                    start = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
                    wav = wav[start:start + self.max_length_samples]
                else:
                    start = max(0, (len(wav) - self.max_length_samples) // 2)
                    wav = wav[start:start + self.max_length_samples]

            return torch.from_numpy(wav).float()
        except Exception as exc:
            print(f"Error loading {audio_path}: {exc}")
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
        audio = self.load_audio(
            row["audio_path"],
            start_time=row["clip_start_seconds"],
            end_time=row["clip_end_seconds"],
        )
        labels = self.get_labels(row)

        return {
            "raw_wav": [audio],
            "text": labels,
            "prompt": self.config.model.prompt_template,
            "task": "species-classification",
            "id": row["audio_path"],
            "index": index,
        }

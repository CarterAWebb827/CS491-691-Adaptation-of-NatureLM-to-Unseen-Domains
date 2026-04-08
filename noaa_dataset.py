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
    # Add the NatureLMaudio directory to Python path
    current_dir = Path.cwd()
    naturelm_dir = Path(os.path.join(current_dir, "NatureLMaudio"))
    if str(naturelm_dir) not in sys.path:
        sys.path.insert(0, str(naturelm_dir))
        print(f"Added {naturelm_dir} to Python path")
    from NatureLMaudio.NatureLM.dataset import collater


class RightWhaleDataset(Dataset):
    SPECIES_NAME = "Right Whale"
    RIWH_CODE = "RIWH"
    LOG_SUFFIX = "_upcall-detection-log.csv"
    METADATA_CACHE_NAME = "metadata_chunks_10s_v2.csv"
    CHUNK_DURATION_SECONDS = 10

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
        self.clip_duration_seconds = self.CHUNK_DURATION_SECONDS
        self.collater = collater

        #create the dataframes and label columns and mark is prepared
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

        metadata_cache = self.root_dir / self.METADATA_CACHE_NAME
        if metadata_cache.exists():
            df = pd.read_csv(metadata_cache)
            for column in ["detection_start_time", "detection_end_time", "chunk_start_time", "chunk_end_time"]:
                if column in df.columns:
                    df[column] = pd.to_datetime(df[column])
            if "output" in df.columns:
                df = df.drop(columns=["output"])
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

            rows.extend(self._build_chunk_records(dataset_dir, log_path, wav_index, detections_df))

        if not rows:
            raise ValueError(f"No usable NOAA rows found in {self.root_dir}")

        df = pd.DataFrame(rows)
        df[self.SPECIES_NAME] = (df["matching_detections"] > 0).astype(int)
        df["task"] = "species-multiple-detection"
        df["instruction"] = "<Audio><AudioHere></Audio> Which of these, if any, are present in the audio recording? North Atlantic Right Whale, North Pacific Right Whale, Southern Right Whale"
        df["dataset_name"] = "noaa-right-whale"

        ordered_columns = ["audio_path", "clip_start_seconds", "clip_end_seconds", "chunk_start_time", "chunk_end_time", "detection_start_time", "detection_end_time", "matching_detections"]
        ordered_columns += ["task", "instruction", self.SPECIES_NAME, "dataset_name", "source_csv", "dataset_dir", "selection", "species_code"]
        ordered_columns += ["detection_confidence", "channel", "low_freq_hz", "high_freq_hz"]
        return df[ordered_columns]

    def _select_log_file(self, data_dir):
        log_files = []

        for path in sorted(data_dir.glob(f"*{self.LOG_SUFFIX}")):
            if "deprecated" in path.name.lower():
                continue
            log_files.append(path)

        if not log_files:
            return None
        return log_files[0]

    def _normalize_detection_log(self, df):
        required_columns = ["Selection", "Channel", "Start_DateTime_ISO8601", "End_DateTime_ISO8601", "Low.Freq..Hz.", "High.Freq..Hz.", "Species", "Detection_Confidence"]

        missing = []
        for column in required_columns:
            if column not in df.columns:
                missing.append(column)

        if missing:
            raise ValueError(f"Missing expected NOAA columns: {missing}")

        df = df[required_columns].copy()
        df = df[df["Species"] == self.RIWH_CODE].copy()
        df["Start_DateTime_ISO8601"] = pd.to_datetime(df["Start_DateTime_ISO8601"], utc=True)
        df["End_DateTime_ISO8601"] = pd.to_datetime(df["End_DateTime_ISO8601"], utc=True)

        # Some logs contain paired Waveform/Spectrogram rows for the same detection.
        df = df.drop_duplicates(subset=["Selection", "Channel", "Start_DateTime_ISO8601", "End_DateTime_ISO8601", "Low.Freq..Hz.", "High.Freq..Hz.", "Species", "Detection_Confidence"])

        return df.reset_index(drop=True)

    def _build_wav_index(self, wav_paths):
        wav_index = []
        for wav_path in wav_paths:
            start_time = self._parse_wav_start_time(wav_path.name)
            duration_seconds = self._get_wav_duration_seconds(wav_path)
            end_time = start_time + timedelta(seconds=duration_seconds)
            wav_index.append({"path": wav_path, "start_time": start_time, "end_time": end_time})
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

    def _build_chunk_records(self, dataset_dir, log_path, wav_index, detections_df):
        detection_records = self._build_detection_records(detections_df)
        rows = []

        for wav_entry in wav_index:
            wav_detections = []
            wav_start_time = wav_entry["start_time"]
            wav_end_time = wav_entry["end_time"]

            for detection in detection_records:
                detection_start_time = detection["start_time"]
                detection_end_time = detection["end_time"]

                if self._times_overlap(detection_start_time, detection_end_time, wav_start_time, wav_end_time):
                    wav_detections.append(detection)

            wav_duration_seconds = (wav_entry["end_time"] - wav_entry["start_time"]).total_seconds()
            chunk_start_seconds = 0.0

            # Evaluate fixed windows so both positive and negative chunks are represented.
            while chunk_start_seconds < wav_duration_seconds:
                chunk_end_seconds = chunk_start_seconds + self.clip_duration_seconds
                if chunk_end_seconds > wav_duration_seconds:
                    chunk_end_seconds = wav_duration_seconds

                chunk_start_time = wav_entry["start_time"] + timedelta(seconds=chunk_start_seconds)
                chunk_end_time = wav_entry["start_time"] + timedelta(seconds=chunk_end_seconds)

                overlapping_detections = []
                for detection in wav_detections:
                    detection_start_time = detection["start_time"]
                    detection_end_time = detection["end_time"]

                    if self._times_overlap(detection_start_time, detection_end_time, chunk_start_time, chunk_end_time):
                        overlapping_detections.append(detection)

                rows.append(self._build_chunk_record(wav_entry=wav_entry, chunk_start_seconds=chunk_start_seconds, chunk_end_seconds=chunk_end_seconds, chunk_start_time=chunk_start_time,
                    chunk_end_time=chunk_end_time, overlapping_detections=overlapping_detections, dataset_dir=dataset_dir, log_path=log_path))

                chunk_start_seconds += self.clip_duration_seconds

        return rows

    def _build_detection_records(self, detections_df):
        records = []

        for _, row in detections_df.iterrows():
            records.append(
                {
                    "selection": int(row["Selection"]),
                    "channel": int(row["Channel"]),
                    "start_time": row["Start_DateTime_ISO8601"].tz_localize(None),
                    "end_time": row["End_DateTime_ISO8601"].tz_localize(None),
                    "species_code": row["Species"],
                    "detection_confidence": row["Detection_Confidence"],
                    "low_freq_hz": float(row["Low.Freq..Hz."]),
                    "high_freq_hz": float(row["High.Freq..Hz."]),
                }
            )

        records.sort(key=lambda record: (record["start_time"], record["end_time"]))
        return records

    def _build_chunk_record(self, wav_entry, chunk_start_seconds, chunk_end_seconds, chunk_start_time, chunk_end_time, overlapping_detections, dataset_dir, log_path):
        if overlapping_detections:
            earliest_detection = overlapping_detections[0]
            detection_confidence = self._select_detection_confidence(overlapping_detections)
            selection = earliest_detection["selection"]
            species_code = earliest_detection["species_code"]
            channel = earliest_detection["channel"]
            detection_start_time = earliest_detection["start_time"]
            detection_end_time = earliest_detection["end_time"]
            low_freq_hz = earliest_detection["low_freq_hz"]
            high_freq_hz = earliest_detection["high_freq_hz"]

            for detection in overlapping_detections[1:]:
                current_start_time = detection["start_time"]
                current_end_time = detection["end_time"]
                current_low_freq_hz = detection["low_freq_hz"]
                current_high_freq_hz = detection["high_freq_hz"]

                if current_start_time < detection_start_time:
                    detection_start_time = current_start_time

                if current_end_time > detection_end_time:
                    detection_end_time = current_end_time

                if current_low_freq_hz < low_freq_hz:
                    low_freq_hz = current_low_freq_hz

                if current_high_freq_hz > high_freq_hz:
                    high_freq_hz = current_high_freq_hz
        else:
            detection_start_time = pd.NaT
            detection_end_time = pd.NaT
            detection_confidence = "None"
            selection = -1
            species_code = ""
            channel = -1
            low_freq_hz = np.nan
            high_freq_hz = np.nan

        return {
            "audio_path": str(wav_entry["path"]),
            "clip_start_seconds": float(chunk_start_seconds),
            "clip_end_seconds": float(chunk_end_seconds),
            "chunk_start_time": chunk_start_time,
            "chunk_end_time": chunk_end_time,
            "detection_start_time": detection_start_time,
            "detection_end_time": detection_end_time,
            "matching_detections": len(overlapping_detections),
            "source_csv": str(log_path),
            "dataset_dir": str(dataset_dir),
            "selection": selection,
            "species_code": species_code,
            "detection_confidence": detection_confidence,
            "channel": channel,
            "low_freq_hz": low_freq_hz,
            "high_freq_hz": high_freq_hz,
        }

    def _select_detection_confidence(self, overlapping_detections):
        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        best_confidence = str(overlapping_detections[0]["detection_confidence"])
        best_rank = confidence_rank.get(best_confidence.strip().lower(), -1)

        for detection in overlapping_detections[1:]:
            current_confidence = str(detection["detection_confidence"])
            current_rank = confidence_rank.get(current_confidence.strip().lower(), -1)

            if current_rank > best_rank:
                best_confidence = current_confidence
                best_rank = current_rank
            elif current_rank == best_rank:
                if current_confidence > best_confidence:
                    best_confidence = current_confidence

        return best_confidence

    def _times_overlap(self, start_a, end_a, start_b, end_b):
        if start_a >= end_b:
            return False

        if end_a <= start_b:
            return False

        return True

    def _sample_dataframe(self, df, percentage):
        if percentage <= 0 or percentage >= 1:
            return df.reset_index(drop=True)
        unique_audio_paths = df["audio_path"].drop_duplicates()
        _, sampled_audio_paths = train_test_split(unique_audio_paths, test_size=percentage, random_state=42)
        sampled_df = df[df["audio_path"].isin(sampled_audio_paths)]
        return sampled_df.reset_index(drop=True)

    def _split_dataframe(self, df):
        unique_audio_paths = df["audio_path"].drop_duplicates()
        train_val_paths, test_paths = train_test_split(unique_audio_paths, test_size=0.1, random_state=42)
        train_paths, val_paths = train_test_split(train_val_paths, test_size=0.2, random_state=42)
        train_df = df[df["audio_path"].isin(train_paths)]
        val_df = df[df["audio_path"].isin(val_paths)]
        test_df = df[df["audio_path"].isin(test_paths)]
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _save_metadata_extra(self, df):
        output_df = df.copy()
        for column in ["detection_start_time", "detection_end_time", "chunk_start_time", "chunk_end_time"]:
            output_df[column] = output_df[column].astype(str)
        output_df.to_csv(self.root_dir / self.METADATA_CACHE_NAME, index=False)

    def load_audio(self, audio_path, start_time=None, end_time=None):
        try:
            if not os.path.exists(audio_path):
                print(f"Warning: audio file not found: {audio_path}")
                return torch.zeros(self.max_length_samples, dtype=torch.float32)

            #set start/stop times
            info = sf.info(audio_path)
            if start_time is None:
                frame_start = 0
            else:
                frame_start = int(float(start_time) * info.samplerate)
                if frame_start < 0:
                    frame_start = 0

            if end_time is None:
                frame_stop = info.frames
            else:
                frame_stop = int(float(end_time) * info.samplerate)
                if frame_stop > info.frames:
                    frame_stop = info.frames

            max_frame_count = int(self.clip_duration_seconds * info.samplerate)
            if frame_stop > frame_start + max_frame_count:
                frame_stop = frame_start + max_frame_count

            if frame_stop <= frame_start:
                frame_stop = frame_start + max_frame_count
                if frame_stop > info.frames:
                    frame_stop = info.frames

            wav, sr = sf.read(audio_path, start=frame_start, stop=frame_stop)

            #reduce to mono
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            #set sample rate to 16000
            if sr != self.sample_rate:
                wav_tensor = torch.from_numpy(wav).float()
                resampler = T.Resample(sr, self.sample_rate)
                wav_tensor = resampler(wav_tensor.unsqueeze(0)).squeeze(0)
                wav = wav_tensor.numpy()

            #pad if needed
            if len(wav) < self.max_length_samples:
                wav = np.pad(wav, (0, self.max_length_samples - len(wav)))
            #trim only trailing spillover so the requested chunk start stays aligned
            elif len(wav) > self.max_length_samples:
                wav = wav[:self.max_length_samples]

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
        audio = self.load_audio(row["audio_path"], start_time=row["clip_start_seconds"], end_time=row["clip_end_seconds"])
        labels = self.get_labels(row)

        return {
            "raw_wav": [audio],
            "text": labels,
            "prompt": self.config.model.prompt_template,
            "task": "species-classification",
            "id": f"{row['audio_path']}:{row['clip_start_seconds']:.2f}-{row['clip_end_seconds']:.2f}",
            "index": index,
        }

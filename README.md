# Adaptation of NatureLM-Audio to Unseen Domains

This repository contains the code and documentation for a research project investigating the effectiveness of the NatureLM-audio model for bioacoustic classification under three distinct paradigms: **zero-shot inference**, **dataset-specific fine-tuning**, and **generalized fine-tuning** across multiple taxonomically diverse benchmark datasets.

## Project Overview

Bioacoustic monitoring is critical for biodiversity conservation, yet the development of robust automated classifiers is often constrained by limited labeled data. This project evaluates NatureLM-audio, a large audio-language model, on three benchmark datasets spanning whales, primates, and anuran amphibians:

| Dataset | Task Type | Taxonomic Group | Classes |
|---------|-----------|-----------------|---------|
| NOAA | Binary detection | Marine mammals | 2 (whale/no whale) |
| Macaque | Call classification | Primates | Call types |
| AnuraSet | Multi-label classification | Anuran amphibians | 42 species |

## Repository Structure

```bash
├── anura_dataset.py        # AnuraSet dataset class
├── anura_zero_shot.py      # Zero-shot evaluation for AnuraSet
├── anura_fine_tune.py      # Fine-tuning script for AnuraSet
├── data/                   # Directory for data storage
├── general_fine_tune.py    # Combined multi-dataset fine-tuning
├── macaque_dataset.py      # Macaque dataset class
├── macaque_zero_shot.py    # Zero-shot evaluation for Macaque
├── NatureLMaudio/          # Directory for NatureLM code
├── noaa_dataset.py         # NOAA Right Whale dataset class
├── noaa_fine_tune.py       # Fine-tuning script for NOAA
├── noaa_zero_shot.py       # Zero-shot evaluation for NOAA
├── setup.ipynb             # Colab setup notebook
├── MacHelper.ipynb         # Macaque data preparation notebook
├── outputs/                # Evaluation results
└── README.md
```

---

## Hardware and Software Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **Operating System** | Linux (Ubuntu 24.04+), macOS 12+, or Windows 10+ with WSL2 |
| **Python** | 3.12 |
| **RAM** | 8 GB minimum |
| **GPU** | AMD RX 7900XT with 20GB of VRAM |
| **Storage** | 35 GB free space for models and datasets |

### Software Dependencies

All experiments use the following key packages with pinned versions (among other needed packages):

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.3.0+ | Deep learning framework |
| `transformers` | 4.44.2 | Hugging Face model hub |
| `peft` | 0.11.1 | Parameter-efficient fine-tuning |
| `bitsandbytes` | 0.43.0+ | 4-bit quantization (QLoRA) |
| `numpy` | 1.26.4 | Numerical computing |
| `pandas` | 2.2.2 | Data manipulation |
| `soundfile` | 0.12.1 | Audio file I/O |
| `librosa` | 0.9.2 | Audio processing |
| `scikit-learn` | 1.1.1 | Machine learning utilities |
| `huggingface-hub` | 0.29.1 | Model download and authentication |

### Pre-trained Models

This project uses the following pre-trained model:

- **NatureLM-audio**: The primary audio-language model from Earth Species Project ([GitHub](https://github.com/earthspecies/NatureLM-audio))

---

## Setup Instructions

Choose one of the following two setup methods depending on your environment.

### Option A: Google Colab Setup (Recommended for Quick Start)

Google Colab provides free GPU access and pre-installed system dependencies, making it the easiest way to get started.

#### Step 1: Mount Google Drive

The notebooks require Google Drive for persistent storage of models and datasets. When prompted, authenticate with your Google account.

#### Step 2: Clone NatureLM-audio to Drive

Before running the notebooks, clone the NatureLM-audio repository to your Google Drive:

```python
# Run this in a Colab cell
import os
%cd /content/drive/MyDrive
!git clone https://github.com/earthspecies/NatureLM-audio.git NatureLMaudio
```

This step only needs to be done once; the model will persist in your Drive.

#### Step 3: Download Dataset(s)

Download the required datasets to your Google Drive:

**NOAA Right Whale:**
Place your NOAA data in `/content/drive/MyDrive/RightWhaleData/` following the structure expected by `noaa_dataset.py`.

**Macaque:**
Use the `MacHelper.ipynb` notebook which automatically downloads and prepares the Macaque dataset.

**AnuraSet:**

```python
# In Colab
%cd /content/drive/MyDrive
!wget https://zenodo.org/records/8342596/files/anuraset.zip?download=1
!unzip AnuraSet.zip
```

#### Step 4: Run a Setup Notebook

Open and run `setup.ipynb` in Google Colab with GPU runtime enabled. This notebook:

1. Installs all system dependencies (libsndfile, ffmpeg, build tools)
2. Installs Python packages with exact pinned versions
3. Installs NatureLM-audio
4. Installs beans-zero for evaluation
5. Installs bitsandbytes for QLoRA quantization
6. Verifies CUDA availability and package versions
7. Clones this repository

### Option B: Local Machine Setup

For users with local GPU hardware, follow these steps.

#### Step 1: System Dependencies

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libavcodec-extra \
    build-essential \
    cmake \
    swig \
    python3-tk \
    libopenblas-dev \
    libomp-dev \
    curl \
    git
```

#### Step 2: Create Python Environment

```bash
# Create and activate conda environment (recommended)
conda create -n naturelm python=3.10
conda activate naturelm

# Or use venv
python -m venv naturelm-env
source naturelm-env/bin/activate # Linux
#      naturelm-env\Scripts\activate # Windows
```

#### Step 3: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Core dependencies
pip install numpy==1.26.4 pandas==2.2.2

# Audio processing
pip install librosa==0.9.2 soundfile==0.12.1 audioread==3.0.1
pip install resampy==0.3.1 pydub==0.25.1

# ML/DL frameworks
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.44.2 accelerate==0.31.0 peft==0.11.1
pip install datasets==3.5.0 einops==0.8.0 safetensors==0.4.3

# Scientific computing
pip install scipy==1.14.0 scikit-learn==1.1.1 matplotlib==3.9.0
pip install bitsandbytes

# Utilities
pip install tqdm==4.66.4 pillow==10.3.0 huggingface-hub==0.29.1
pip install pyyaml==6.0 rich==14.0.0 pydantic==2.7.4
pip install python-dotenv==1.0.1 numba==0.60.0 llvmlite==0.43.0
pip install openpyxl wandb tensorboard seaborn

# Audio evaluation
pip install mir-eval==0.7
```

#### Step 4: Install NatureLM-audio

```bash
git clone https://github.com/earthspecies/NatureLM-audio.git
cd NatureLM-audio
pip install -e .[gpu] --no-deps
cd ..
```

#### Step 5: Install beans-zero

```bash
pip install --no-deps git+https://github.com/earthspecies/beans-zero.git@31d4487ee6452ae6c31853d45fd38b7d4150372d
```

#### Step 6: Clone This Repository

```bash
git clone https://github.com/CarterAWebb827/CS491-691-Adaptation-of-NatureLM-to-Unseen-Domains.git
cd CS491-691-Adaptation-of-NatureLM-to-Unseen-Domains
```

#### Step 7: Download Datasets

Create a `data/` directory and download datasets:

```bash
mkdir -p data

# AnuraSet
wget https://zenodo.org/records/8342596/files/anuraset.zip?download=1 -P data/
unzip data/AnuraSet.zip -d data/AnuraSet

# Macaque (automatic via script)
python -c "from macaque_dataset import MacaqueDataset; print('Dataset class ready')"

# NOAA - place your data in data/NOAA/
```

#### Step 8: Verify Installations

```bash
python -c "
import torch
import transformers
import bitsandbytes
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
print(f'bitsandbytes: {bitsandbytes.__version__}')
"
```

---

## Hugging Face Authentication

NatureLM-audio requires Hugging Face authentication to download model weights. Set your token:

```bash
# Option 1: Environment variable (recommended for scripts)
export HF_TOKEN="your_huggingface_token"

# Option 2: Login interactively
python -c "from huggingface_hub import login; login()"
```

Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Running Experiments

### Zero-Shot Evaluation

Zero-shot evaluation requires no training—it directly evaluates the pre-trained NatureLM-audio model.

#### AnuraSet Zero-Shot

```bash
python anura_zero_shot.py
```

**Expected runtime:** ~2–4 hours on L4 GPU for full dataset

#### NOAA Zero-Shot

```bash
python noaa_zero_shot.py
```

**Expected runtime:** ~3-5 hours on L4 GPU

#### Macaque Zero-Shot

```bash
python macaque_zero_shot.py
```

**Expected runtime:** ~1-3 minutes on L4 GPU

### Dataset-Specific Fine-Tuning

Fine-tuning uses LoRA (Low-Rank Adaptation) with QLoRA 4-bit quantization for memory efficiency.

#### AnuraSet Fine-Tuning

```bash
python anura_fine_tune.py
```

**Hyperparameter Configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--learning_rate` | `1e-6` | Initial learning rate for AdamW optimizer |
| `--warmup_steps` | `200` | Linear warmup steps |
| `--weight_decay` | `0.01` | Weight decay for regularization |
| `--max_epochs` | `15` | Maximum training epochs |
| `--batch_size` | `1` | Per-device batch size |
| `--accum_grad_iters` | `4` | Gradient accumulation steps (effective batch = batch_size * accum_grad_iters) |
| `--lora_rank` | `8` | LoRA decomposition rank |
| `--lora_alpha` | `32` | LoRA scaling factor |
| `--use_4bit` | `True` | Enable 4-bit quantization (QLoRA) |
| `--bnb_4bit_compute_dtype` | `bfloat16` | Compute precision for quantized layers |
| `--bnb_4bit_quant_type` | `nf4` | Quantization data type |
| `--use_gradient_checkpointing` | `True` | Trade compute for memory savings |
| `--clear_cache_every_n_steps` | `10` | CUDA cache clearing frequency |
| `--random_seed` | `42` | Random seed for reproducibility |

**Expected runtime:** 6 days on L4 GPU for full dataset;

#### NOAA Fine-Tuning

```bash
python noaa_fine_tune.py
```

**Expected runtime:** 8-9 days on L4 GPU

### Generalized (Combined) Fine-Tuning

Train a single model on all three datasets simultaneously:

```bash
python general_fine_tune.py
```

**Dataset percentage arguments (for controlled experiments):**

- `--use_percentage FLOAT`: Apply to all datasets uniformly
- `--macaque_percentage FLOAT`: Macaque-specific subset
- `--noaa_percentage FLOAT`: NOAA-specific subset
- `--anura_percentage FLOAT`: AnuraSet-specific subset

---

## Reproducibility

All experiments use fixed random seeds of 42 to ensure reproducibility:

To reproduce exact results:

1. Use the pinned package versions listed in the dependencies section
2. Set `--random_seed 42` in all fine-tuning scripts
3. Use `--use_predefined_splits` (default) for consistent train/val/test partitions

---

## Output Structure

Results are organized as follows:

```bash
outputs/
├── anura_zeroshot/
│   ├── zero_shot_test_results.txt        # Raw model outputs
│   ├── zero_shot_test_summary.txt        # Evaluation summary with metrics
├── anura_finetune/
│   ├── anura_finetune_lora8_lr0.0001/
│   │   ├── checkpoint_best.pth            # Best model checkpoint
│   │   ├── fine_tune_test_results.txt
│   │   ├── fine_tune_test_summary.txt
├── noaa_zeroshot/
│   └── ...
├── noaa_finetune/
│   └── ...
├── macaque_zeroshot/
│   └── ...
└── combined_finetune/
    ├── combined_finetune_lora8_lr0.0001/
    │   ├── checkpoint_best.pth
    │   └── combined_fine_tune_test_summary.txt  # Per-dataset breakdown
```

---

## Key Results Summary

| Dataset | Paradigm | Primary Metric | Precision | Recall | F1 |
|---------|----------|----------------|-----------|--------|-----|
| Macaque | Zero-shot | 66.89% accuracy | 1.000 | 0.669 | 0.802 |
| Macaque | Specific FT | 100.00% accuracy | 1.000 | 1.000 | 1.000 |
| Macaque | generalized FT | 100.00% accuracy | 1.000 | 1.000 | 1.000 |
| NOAA | Zero-shot | 91.89% accuracy | 0.184 | 0.026 | 0.045 |
| NOAA | Specific FT | 92.55% accuracy | 0.000 | 0.000 | 0.000 |
| NOAA | generalized FT | 90.37% accuracy | 0.000 | 0.000 | 0.000 |
| AnuraSet | Zero-shot | 39.33% accuracy | 0.849 | 0.086 | 0.157 |
| AnuraSet | Specific FT | 82.71% accuracy | 0.942 | 0.906 | 0.924 |
| AnuraSet | generalized FT | 76.54% accuracy | — | — | — |

---

## Troubleshooting

### Common Issues

**CUDA out of memory:**

- Reduce `--batch_size` (try 4 or 2)
- Increase `--accum_grad_iters` to maintain effective batch size
- Enable `--use_4bit` and `--use_gradient_checkpointing` if you haven't

**Dataset not found:**

- Verify data directory structure matches dataset class expectations
- Use absolute paths for `--data_dir` arguments

**Package version conflicts:**

- Create a fresh conda/venv environment
- Install packages in the order specified above
- Use `pip install --no-deps` for problematic packages

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@misc{adaptation-naturelm-unseen-domains,
  author = {Webb, Carter and Wilfong, Jessi and Hogueison, Jennifer},
  title = {Adaptation of NatureLM-Audio to Unseen Domains for Bioacoustic Classification},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/CarterAWebb827/CS491-691-Adaptation-of-NatureLM-to-Unseen-Domains}
}
```

Also cite the NatureLM-audio model:

```bibtex
@article{robinson2024naturelm,
  title={NatureLM-audio: A large audio-language model for biodiversity monitoring},
  author={Robinson, David and Lappan, Sara and Kahl, Stefan and Klinck, Holger},
  journal={arXiv preprint arXiv:2403.12345},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License. See the NatureLM-audio repository for the base model's license terms.

---

## Acknowledgments

- **NatureLM-audio**: Earth Species Project for the pre-trained model and codebase
- **AnuraSet**: Cañas et al. for the anuran bioacoustics benchmark and dataset
- **NOAA Right Whale**: Dugan et al. for the marine mammal dataset
- **Macaque**: Earth Species Project for the primate vocalization dataset
- **CS 491/691**: Course instructor for guidance and feedback

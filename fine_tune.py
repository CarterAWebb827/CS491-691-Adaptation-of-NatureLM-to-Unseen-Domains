import os
import gc
from pathlib import Path
from huggingface_hub import HfFolder, login
import argparse

token = HfFolder.get_token()
login(token=token)
del token
gc.collect()

current_dir = Path.cwd()
naturelm_dir = Path(os.path.join(current_dir, "NatureLMaudio"))
# sys.path.append(naturelm_dir) # Appending to system path allows us to do "import infer" instead of "import NatureLMaudio.infer"

from NatureLMaudio.NatureLM.config import Config
from NatureLMaudio.NatureLM.infer import load_model_and_config
from NatureLMaudio.NatureLM.runner import Runner

from anura_dataset import AnuraDataset

def get_anura_datasets(config, percentage, data_dir):
    datasets = {}

    datasets["train"] = AnuraDataset(config=config, percentage=percentage, split="train", data_dir=data_dir)
    datasets["valid"] = AnuraDataset(config=config, percentage=percentage, split="valid", data_dir=data_dir)
    datasets["test"] = AnuraDataset(config=config, percentage=percentage, split="test", data_dir=data_dir)

    return datasets

def main():
    parser = argparse.ArgumentParser(description="A script to fine-tune the NatureLM-audio model on frog and toad species classification")
    parser.add_argument("--percentage", type=float, default=None, help="Designate the percentage of the full dataset used for fine-tuning")
    parser.add_argument("--naturelm_dir", type=str, default="NatureLMaudio", help="Designate the location of the NatureLM-audio directory")
    parser.add_argument("--data_dir", type=str, default="data/AnuraSet", help="Designate the location of the data directory to be used")
    args = parser.parse_args()

    # Load our config
    cfg_path = "NatureLMaudio/configs/finetune_anura.yaml"
    cfg = Config.from_sources(cfg_path)

    # Create job ID for the runner naming convention
    job_id = f"anura_finetune_lora{cfg.model.lora_rank}_lr{cfg.run.optims.init_lr}"

    # Load the base model
    print("Loading the model...")
    model, _ = load_model_and_config(cfg_path=cfg_path, device=cfg.model.device)

    # Configure the LoRA
    model.lora = cfg.model.lora
    model.lora_rank = cfg.model.lora_rank
    model.lora_alpha = cfg.model.lora_alpha

    # Make sure only LoRA parameters are trainable
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Prepare the datasets
    print("Preparing datasets...")
    datasets = get_anura_datasets(cfg, args.percentage, args.data_dir)

    # Initialize the runner
    print("Initializing runner...")
    runner = Runner(cfg, model, datasets, job_id)

    # Start training
    print("Starting training...")
    runner.train()

if __name__ == "__main__":
    main()
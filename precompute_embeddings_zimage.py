#!/usr/bin/env python3
"""
Precompute text embeddings for Z-Image-Turbo training

This script computes and saves text encoder embeddings to avoid OOM during training.
Qwen3Model is too large to fit in GPU memory together with other models.
"""

import argparse
import logging
import os
from pathlib import Path

# Set HuggingFace mirror BEFORE importing huggingface libraries
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute text embeddings for training")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pbrt_dataset",
        help="The name of the Dataset to train on",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="default",
        help="The configuration of the dataset to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/embeddings_zimage",
        help="The output directory where the embeddings will be written",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for computing embeddings",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for dataset loading",
    )
    args = parser.parse_args()
    return args


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, variant: str
):
    text_encoder_config = None
    text_encoder_cls = None

    # Try to load from transformers library
    from transformers import AutoConfig

    text_encoder_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "Qwen3ForCausalLM":
        from transformers import Qwen3Model

        return Qwen3Model
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples):
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"input_ids": input_ids}


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Load tokenizer
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {args.pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # Load text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, args.variant
    )

    # Load text encoder
    logger.info(f"Loading text encoder: {text_encoder_cls.__name__}")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )

    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    logger.info(f"Text encoder loaded on {device}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    # Tokenize captions
    def tokenize_captions(examples):
        captions = []
        for caption in examples["prompt"]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(caption[0])  # Use first caption for consistency
            else:
                raise NotImplementedError
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Return dict for dataset.map
        return {"input_ids": inputs.input_ids}

    # Apply tokenization
    dataset = dataset.map(
        tokenize_captions,
        batched=True,
        remove_columns=["prompt"],
        desc="Tokenizing captions",
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute embeddings
    logger.info("Computing text embeddings...")
    all_embeddings = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing embeddings")):
            input_ids = batch["input_ids"].to(device)

            # Get text embeddings
            encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

            # Move to CPU and convert to float32 for storage
            encoder_hidden_states = encoder_hidden_states.cpu().float().numpy()

            all_embeddings.append(encoder_hidden_states)

    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    logger.info(f"Computed embeddings shape: {all_embeddings.shape}")

    # Save embeddings
    output_file = output_dir / "text_embeddings.npy"
    np.save(output_file, all_embeddings)
    logger.info(f"Saved embeddings to {output_file}")

    # Save metadata
    metadata = {
        "num_samples": all_embeddings.shape[0],
        "embedding_dim": all_embeddings.shape[1],
        "seq_length": all_embeddings.shape[2],
        "model_name": args.pretrained_model_name_or_path,
    }

    import json

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_file}")
    logger.info("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)

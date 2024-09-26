# embedding_generator.py

import os
import argparse
from dataclasses import dataclass
from typing import Optional
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from gpt import (
    load_model_from_checkpoint,
    generate,
    preprocess_tokens_from_huggingface,
    GPTConfig
)


@dataclass
class EmbeddingGeneratorConfig:
    batch_size: int = 512
    block_size: int = 512
    n_embd: int = 512
    ratio_tokens_saved: float = 0.07
    residual_layer: int = 6
    mb_per_save: int = 2000
    save_dir: str = "./residuals/"


class EmbeddingGenerator:
    def __init__(
        self,
        model_checkpoint: str,
        dataset_dir: str,
        tokenizer_name: str = "activated-ai/tiny-stories-8k-tokenizer",
        config: Optional[EmbeddingGeneratorConfig] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(self.device)
        assert self.device == "cuda", "This module is optimized for GPU (CUDA)."

        torch.autograd.set_grad_enabled(False)

        self.model = load_model_from_checkpoint(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        preprocess_tokens_from_huggingface(dataset_dir)
        self.train_data = torch.load(
            os.path.join(dataset_dir, "train.pt"), map_location=self.device
        ).long()

        if config is None:
            self.config = EmbeddingGeneratorConfig(
                batch_size=512,
                block_size=self.model.config.block_size,
                n_embd=self.model.config.n_embd,
                residual_layer=round(self.model.config.n_layer * 0.65),
            )
        else:
            self.config = config

        self.batches = self._prepare_batches()

    def _prepare_batches(self):
        dataset_size = self.train_data.size(0)
        tokens_per_batch = self.config.batch_size * self.config.block_size
        dataset_remainder = dataset_size % tokens_per_batch
        dataset_length = dataset_size - dataset_remainder

        if dataset_remainder != 0:
            print(f"Removed {dataset_remainder} tokens to make data evenly divisible.")

        return self.train_data[:dataset_length].view(
            -1, self.config.batch_size, self.config.block_size
        )

    def _n_embd_to_mb(self, n_embeddings: int) -> float:
        mb_per_embedding = self.config.n_embd * 2 / 1_000_000  # 2 bytes per bf16
        return mb_per_embedding * n_embeddings

    def _should_save(self, num_embeddings: int) -> bool:
        return self._n_embd_to_mb(num_embeddings) > self.config.mb_per_save

    def generate_embeddings(self):
        os.makedirs(self.config.save_dir, exist_ok=True)

        save_residuals_buffer = []
        token_indices_buffer = []
        context_window_starts_buffer = []
        save_counter = 0

        total_embeddings = int(
            self.batches.size(0)
            * self.config.ratio_tokens_saved
            * self.config.block_size
            * self.config.batch_size
        )
        estimated_storage = self._n_embd_to_mb(total_embeddings)
        print(f"Estimated storage on disk: {estimated_storage:.2f} MB")

        for batch_index, batch in enumerate(tqdm(self.batches)):
            tokens_per_batch = self.config.batch_size * self.config.block_size
            global_token_start = batch_index * tokens_per_batch
            num_tokens_to_save = int(tokens_per_batch * self.config.ratio_tokens_saved)

            local_indices = torch.randperm(tokens_per_batch)[:num_tokens_to_save]
            global_indices = local_indices + global_token_start
            context_window_starts = global_indices - global_indices % self.config.block_size

            token_indices_buffer.extend(global_indices.tolist())
            context_window_starts_buffer.extend(context_window_starts.tolist())

            model_output = self.model(
                batch, return_layer_embs=self.config.residual_layer
            ).view(-1, self.config.n_embd)[local_indices, :]

            save_residuals_buffer.append(model_output)

            num_embeddings_in_buffer = len(save_residuals_buffer) * num_tokens_to_save
            if self._should_save(num_embeddings_in_buffer):
                self._save_embeddings(
                    save_residuals_buffer,
                    token_indices_buffer,
                    context_window_starts_buffer,
                    save_counter,
                )
                save_counter += 1
                save_residuals_buffer.clear()
                token_indices_buffer.clear()
                context_window_starts_buffer.clear()

        # Save any remaining embeddings
        if save_residuals_buffer:
            self._save_embeddings(
                save_residuals_buffer,
                token_indices_buffer,
                context_window_starts_buffer,
                save_counter,
            )

    def _save_embeddings(
        self,
        residuals_buffer,
        token_indices_buffer,
        context_window_starts_buffer,
        file_index: int,
    ):
        residuals_tensor = torch.cat(residuals_buffer).to(torch.bfloat16)
        save_path = os.path.join(self.config.save_dir, f"{file_index}.pt")
        torch.save(
            {
                "residuals": residuals_tensor,
                "token_idxs": token_indices_buffer,
                "token_values": self.train_data[token_indices_buffer],
                "context_window_starts": context_window_starts_buffer,
                "config": self.config,
            },
            save_path,
        )

    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 1.0):
        return generate(self.model, self.tokenizer, prompt, max_length, temperature)

    def load_residual(self, file_index: int):
        file_path = os.path.join(self.config.save_dir, f"{file_index}.pt")
        return torch.load(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Generator Script")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="./llms/50m_llm.pt",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./datasets",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="activated-ai/tiny-stories-8k-tokenizer",
        help="Name of the tokenizer",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--ratio_tokens_saved",
        type=float,
        default=0.07,
        help="Ratio of tokens to save per batch",
    )
    parser.add_argument(
        "--residual_layer",
        type=int,
        help="Layer from which to extract residuals (default: 65% of total layers)",
    )
    parser.add_argument(
        "--mb_per_save",
        type=int,
        default=2000,
        help="Memory buffer size (in MB) before saving to disk",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./residuals/",
        help="Directory to save residual embeddings",
    )
    args = parser.parse_args()

    # Load the model to get block_size and n_embd
    model = load_model_from_checkpoint(args.model_checkpoint)

    # Determine residual layer if not provided
    if args.residual_layer is None:
        args.residual_layer = round(model.config.n_layer * 0.65)

    config = EmbeddingGeneratorConfig(
        batch_size=args.batch_size,
        block_size=model.config.block_size,
        n_embd=model.config.n_embd,
        ratio_tokens_saved=args.ratio_tokens_saved,
        residual_layer=args.residual_layer,
        mb_per_save=args.mb_per_save,
        save_dir=args.save_dir,
    )

    generator = EmbeddingGenerator(
        model_checkpoint=args.model_checkpoint,
        dataset_dir=args.dataset_dir,
        tokenizer_name=args.tokenizer_name,
        config=config,
    )

    generator.generate_embeddings()

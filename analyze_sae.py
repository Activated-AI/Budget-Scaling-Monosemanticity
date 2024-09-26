# sae_processing.py

import torch
import os
from tqdm import tqdm
from sae import TopKSAE
from generate_residuals import EmbeddingGeneratorConfig
import transformers
import argparse

class SAEProcessor:
    def __init__(self, sae_model_path, residuals_dir, tokenizer_name='activated-ai/tiny-stories-8k-tokenizer', dataset_dir='./datasets', device=None):
        """
        Initialize the SAE Processor.

        Args:
            sae_model_path (str): Path to the saved SAE model.
            residuals_dir (str): Directory containing residuals files.
            tokenizer_name (str): Name of the tokenizer to use.
            dataset_dir (str): Directory containing the dataset.
            device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to None.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        torch.set_default_device(self.device)
        assert self.device == 'cuda', "This module is not optimized for CPU"

        torch.autograd.set_grad_enabled(False)

        # Load the SAE model
        self.load_sae_model(sae_model_path)

        # Set residuals directory
        self.residuals_dir = residuals_dir

        # Load the tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

        # Preprocess tokens
        from gpt import preprocess_tokens_from_huggingface
        preprocess_tokens_from_huggingface(dataset_dir)

        # Load the dataset
        self.train_data = torch.load(os.path.join(dataset_dir, "train.pt"), map_location=self.device).to(torch.int64)

    def load_sae_model(self, sae_model_path):
        """
        Load the SAE model from the given path.

        Args:
            sae_model_path (str): Path to the saved SAE model.
        """
        s = torch.load(sae_model_path)
        self.sae = TopKSAE(s['config'])
        self.sae.load_state_dict(s['model'])

    def crop_fit_batch(self, tensor, batch_size):
        """
        Crop the tensor to fit into batches of given size.

        Args:
            tensor (torch.Tensor): Input tensor.
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Cropped tensor.
        """
        n = tensor.size(0)
        n_batches = n // batch_size
        n = n_batches * batch_size
        return tensor[:n]

    def process_residuals(self, residuals_path):
        """
        Process residuals file and get top-k indices and strengths.

        Args:
            residuals_path (str): Path to the residuals file.

        Returns:
            topk_idxs (torch.Tensor): Top-k indices per token.
            topk_strengths (torch.Tensor): Top-k strengths per token.
            token_idxs (list): List of token indices.
            context_window_starts (list): List of context window start indices.
        """
        topk_idxs_per_token = []
        topk_strengths_per_token = []

        residual = torch.load(residuals_path)
        residuals = residual['residuals'].to(torch.float32)
        token_idxs = residual['token_idxs']
        context_window_starts = residual['context_window_starts']
        config = residual['config']
        self.sae.eval()
        cropped = self.crop_fit_batch(residuals, self.sae.config.batch_size)
        batches = cropped.view(-1, self.sae.config.batch_size, self.sae.config.embedding_size)
        with torch.no_grad():
            for batch in tqdm(batches, desc="Processing batches"):
                sae_out = self.sae(batch)
                topk_idxs = sae_out['topk_idxs']
                topk_values = sae_out['topk_values']
                topk_idxs_per_token.append(topk_idxs)
                topk_strengths_per_token.append(topk_values)
            topk_idxs = torch.cat(topk_idxs_per_token, dim=0)
            topk_strengths = torch.cat(topk_strengths_per_token, dim=0)
        return topk_idxs, topk_strengths, token_idxs, context_window_starts

    def get_feature_tokens(self, topk_idxs_per_token, topk_strengths_per_token, feature_idx, top_n=10):
        """
        Get tokens associated with a given feature index.

        Args:
            topk_idxs_per_token (torch.Tensor): Top-k indices per token.
            topk_strengths_per_token (torch.Tensor): Top-k strengths per token.
            feature_idx (int): The feature index to search for.
            top_n (int): Number of top tokens to return.

        Returns:
            List of token indices corresponding to the feature.
        """
        flat_topk_idxs = topk_idxs_per_token.view(-1)
        feature_indexes = torch.where(flat_topk_idxs == feature_idx)[0]
        feature_strengths = topk_strengths_per_token.view(-1)[feature_indexes]

        if feature_strengths.numel() == 0:
            print(f"No occurrences of feature {feature_idx} found.")
            return []

        subset_strengths, subset_flat_feature_idxs = feature_strengths.topk(min(top_n, feature_strengths.numel()))
        # Map back to the global indices
        subset_flat_feature_idxs = feature_indexes[subset_flat_feature_idxs]

        # Get the token indices
        subset_token_idxs = subset_flat_feature_idxs // self.sae.config.topk

        return subset_token_idxs

    def get_context(self, token_idx, window_size=10):
        """
        Get context around a given token index.

        Args:
            token_idx (int): Index of the token.
            window_size (int): Number of tokens to include before and after.

        Returns:
            str: Context string with the token highlighted.
        """
        start = max(token_idx - window_size, 0)
        end = min(token_idx + window_size + 1, len(self.train_data))
        context_tokens = self.train_data[start:end]
        token_position = token_idx - start
        context_tokens = context_tokens.tolist()

        # Insert markers around the token of interest
        context_tokens.insert(token_position + 1, self.tokenizer.convert_tokens_to_ids('>'))  # add after token
        context_tokens.insert(token_position, self.tokenizer.convert_tokens_to_ids('<'))      # add before token

        # Convert tokens to text
        context_text = self.tokenizer.decode(context_tokens, clean_up_tokenization_spaces=False)
        return context_text

    def get_residuals_files(self):
        """
        Get list of residuals files.

        Returns:
            List of residuals file paths.
        """
        residuals_files = [os.path.join(self.residuals_dir, f) for f in os.listdir(self.residuals_dir) if f.endswith('.pt')]
        residuals_files.sort()
        return residuals_files

def main():
    parser = argparse.ArgumentParser(description='Process SAE residuals and extract token contexts.')
    parser.add_argument('--sae_model_path', type=str, required=True, help='Path to the saved SAE model.')
    parser.add_argument('--residuals_dir', type=str, required=True, help='Directory containing residuals files.')
    parser.add_argument('--tokenizer_name', type=str, default='activated-ai/tiny-stories-8k-tokenizer', help='Name of the tokenizer to use.')
    parser.add_argument('--dataset_dir', type=str, default='./datasets', help='Directory containing the dataset.')
    parser.add_argument('--device', type=str, default=None, help="Device to use ('cuda' or 'cpu').")
    parser.add_argument('--residuals_file', type=str, default=None, help='Specific residuals file to process. If not provided, the latest one is used.')
    parser.add_argument('--feature_idx', type=int, default=None, help='Feature index to analyze. If not provided, a random feature is selected.')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top tokens to display for the feature.')
    parser.add_argument('--window_size', type=int, default=10, help='Context window size around the token.')
    args = parser.parse_args()

    # Initialize the processor
    processor = SAEProcessor(
        sae_model_path=args.sae_model_path,
        residuals_dir=args.residuals_dir,
        tokenizer_name=args.tokenizer_name,
        dataset_dir=args.dataset_dir,
        device=args.device
    )

    # Get the residuals files
    residuals_files = processor.get_residuals_files()

    # Select the residuals file to process
    if args.residuals_file is not None:
        residuals_path = args.residuals_file
    else:
        residuals_path = residuals_files[-1]  # Use the latest residuals file

    print(f"Processing residuals file: {residuals_path}")

    # Process residuals
    topk_idxs_per_token, topk_strengths_per_token, token_idxs, context_window_starts = processor.process_residuals(residuals_path)

    # Select a feature index
    if args.feature_idx is not None:
        feature_idx = args.feature_idx
    else:
        # Select a random feature index
        feature_idx = topk_idxs_per_token.view(-1)[torch.randint(high=topk_idxs_per_token.numel(), size=(1,))].item()
        print(f"No feature index provided. Randomly selected feature index: {feature_idx}")

    print(f"Analyzing feature index: {feature_idx}")

    # Get tokens associated with this feature
    feature_token_idxs = processor.get_feature_tokens(topk_idxs_per_token, topk_strengths_per_token, feature_idx, top_n=args.top_n)

    if feature_token_idxs.numel() == 0:
        print("No tokens associated with the selected feature index.")
        return

    # Convert token_idxs to tensor if not already
    if not isinstance(token_idxs, torch.Tensor):
        token_idxs = torch.tensor(token_idxs)

    # Print contexts for these tokens
    for idx in feature_token_idxs:
        token_idx = idx.item()
        dataset_token_idx = token_idxs[token_idx].item()
        context = processor.get_context(dataset_token_idx, window_size=args.window_size)
        print("-" * 40)
        print(context)
        print("-" * 40)

if __name__ == '__main__':
    main()

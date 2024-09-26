import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from dataclasses import dataclass

def r2_per_channel(predicted, actual):
    assert len(predicted.shape) != 1, "You can't compute R² with one sample!"
    channel_means = torch.mean(actual, dim=-2)
    avg_squared_error_per_channel = torch.mean((actual - channel_means) ** 2, dim=-2)
    avg_squared_error_predicted = torch.mean((predicted - actual) ** 2, dim=-2)
    return 1 - avg_squared_error_predicted / avg_squared_error_per_channel

@dataclass
class TopKSAEConfig:
    embedding_size: int = 512
    n_features: int = 32768
    topk: int = 24
    lr: int = 1e-3
    batch_size: int = 32


class TopKSAE(nn.Module):
    def __init__(self, config: TopKSAEConfig):
        super(TopKSAE, self).__init__()
        self.config = config

        self.embedding_size = config.embedding_size
        self.n_features = config.n_features
        self.topk = config.topk

        self.encode = nn.Linear(self.embedding_size, self.n_features, bias=True)
        self.decode = nn.Linear(self.n_features, self.embedding_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.embedding_size))

        # Initialize encode weights with L2 norm randomly between 0.05 and 1.0
        direction_lengths = torch.rand(self.n_features) * 0.95 + 0.05
        direction_lengths = direction_lengths.unsqueeze(-1)
        self.encode.weight.data = F.normalize(self.encode.weight.data, p=2, dim=-1) * direction_lengths

        # Initialize decode weights as the transpose of encode weights
        self.decode.weight.data = self.encode.weight.data.t()

    def keep_topk(self, tensor, k):
        values, indices = torch.topk(tensor, k)
        mask = torch.zeros_like(tensor)
        mask.scatter_(-1, indices, 1)
        result = tensor * mask
        return result, values, indices

    def forward(self, x, return_r2=False):
        x = x - self.bias
        encoded = self.encode(x)
        encoded, values, indices = self.keep_topk(encoded, self.topk)
        decoded = self.decode(encoded) + self.bias
        mse = torch.mean((x - decoded) ** 2)

        output = {
            "encoded": encoded,
            "decoded": decoded,
            "mse": mse,
            "topk_idxs": indices,
            "topk_values": values,
        }

        if return_r2:
            r2 = r2_per_channel(decoded, x)
            mean_r2 = torch.mean(r2)
            output["r2"] = r2
            output["mean_r2"] = mean_r2

        return output

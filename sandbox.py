import math

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
from torch import nn

def generate_dataset(n_samples, noise, return_numpy, visualise_data):
    """Generates moon dataset. Then converts to a torch tensor. Finally normalises the dataset with Z score normalisation for a mean of zero and standard deviation of 1. 

    Args:
        n_samples (int): The number of samples in the moon dataset.
        noise (float): The amount of noise to inject into the dataset.
        return_numpy (bool): Whether to return numpy array.
        visualise_data (bool): Whether to visualise the dataset created.

    Returns:
        (torch.Tensor or numpy array) Return a torch tensor or numpy array."""
    data, labels = make_moons(n_samples=n_samples, noise=noise)

    x1 = torch.tensor(data, dtype=torch.float32)
    x1 = (x1 - x1.mean(dim=0)) / x1.std(dim=0)

    if visualise_data:
        x_np = x1.numpy()
        plt.scatter(x_np[:, 0], x_np[:, 1], c=labels, cmap='coolwarm', s=5)
        plt.title("Moons Dataset (colored by class)")
        plt.show()

    if return_numpy:
        x_np = x1.numpy()
        return x_np

    return x1

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [batch, 1] -> converts to a vector of size 'dim'
        device = t.device
        half_dim = self.dim // 2 # Dimension is the size of the embedding vector. We will assign half embedding vectors to cosine and others to sine.
        scaling_factor = math.log(10000) / (half_dim - 1) # scaling factor for each frequency log[1, 1/10, 1/100 ...], logarithmically
        embeddings = torch.arange(half_dim, device=device) * -scaling_factor# Embedding vector combined with scaling factor
        embeddings = torch.exp(embeddings) # add exponential to remove log -[0, 1/10, 2/100...]
        embeddings = embeddings[None, :] * t# add time
        embeddings = torch.cat((embeddings.sine(), embeddings.cosine()), dim=-1), # add cosine and sine and combine together
        return embeddings
    
class FlowMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, time_dim=64):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim), # input dim (x, y) + dimensions of time sinusoidal embedding
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):

        if len(t.shape) == 1:
            t = t.unsqueeze(-1) # we want our input value to be of shape 2 always, so only add another dimension if it only has one dimension.
        
        t_emb = self.time_mlp(t)

        x_input = torch.cat([x, t_emb], dim=1)

        velocity = self.net(x_input)
        return velocity
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowMLP().to(device)

    print(f"Model initialized on: {device}")
import math

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
from torch import nn
from torch import optim

def generate_dataset(n_samples, noise, return_numpy=False, visualise_data=False):
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
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # add cosine and sine and combine together
        return embeddings
    
class FlowMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=512, time_dim=128):
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
        
        t_emb = self.time_mlp(t * 1000)

        x_input = torch.cat([x, t_emb], dim=1)

        velocity = self.net(x_input)
        return velocity 
    

def train_step(model, optimizer, x1, device):

    optimizer.zero_grad()

    x0 = torch.randn_like(x1) # Sample random points of batch size for x1. Use torch.randn_link(x1) for noise sampled from Gaussian distribution.
    batch = x1.shape[0]
    t = torch.rand((batch, 1), device=device) # Sample time values between 1 and 0 of size batch size. Use torch.rand instead for torch.randn for smooth distribution.

    t = t * (1 - 1e-3) + 1e-3

    xt = (1 - t) * x0 + t * x1 # value of x between x0 and x1.

    # eps = 1e-3
    # t = torch.rand((batch, 1), device=device)
    # t = t * (1 - eps) + eps

    # Stable version
    target_velocity = x1 - x0

    # target_velocity = (x1 - xt) / (1 - t + 1e-5)

    v_pred = model(xt, t)
    loss = torch.mean((v_pred - target_velocity) ** 2)

    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def sample(model, n_samples=1000, steps=300, device='cpu'):
    model.eval()
    # 1. Start with pure Gaussian noise (x0)
    x = torch.randn((n_samples, 2)).to(device)
    dt = 1.0 / steps
    
    # 2. Iteratively push the noise along the predicted velocity vectors
    for i in range(steps):
        # Current time t from 0 to 1
        t = torch.full((n_samples, 1), i / steps).to(device)
        v = model(x, t)
        x = x + v * dt # Euler integration step
        
    model.train()
    return x.cpu().numpy()


# --- Visualization Helpers ---
def plot_results(model, x1_data, device):
    generated = sample(model, n_samples=2000, steps=100, device=device)
    real = x1_data[:2000].cpu().numpy()

    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Overlay
    plt.subplot(1, 2, 1)
    plt.scatter(real[:, 0], real[:, 1], color='blue', alpha=0.2, s=5, label='Real')
    plt.scatter(generated[:, 0], generated[:, 1], color='red', alpha=0.5, s=5, label='Flow')
    plt.title("Overlay: Real vs. Flow")
    plt.legend()

    # Subplot 2: Vector Field at t=0.5
    plt.subplot(1, 2, 2)
    grid = torch.linspace(-3, 3, 20)
    gx, gy = torch.meshgrid(grid, grid, indexing='ij')
    points = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(device)
    # Use points.shape to get the integer count of points
    t_half = torch.full((points.shape[0], 1), 0.5).to(device)
    with torch.no_grad():
        v = model(points, t_half).cpu()
    plt.quiver(
        points[:, 0].cpu(),
        points[:, 1].cpu(),
        v[:, 0],
        v[:, 1],
        color='green'
    )
    plt.title("Velocity Field (t=0.5)")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Model initialized on: {device}")
    print(torch.cuda.get_device_name(0))

    num_samples = 5000
    noise_ratio = 0.05
    batch_size = 256

    x1_data = generate_dataset(num_samples, noise_ratio).to(device)
    print(f"Moons dataset generated | Number of samples: {num_samples} | Noise ratio: {noise_ratio}")

    print(f"Commencing training with batch size: {batch_size}...")
    for epoch in range(20001):
        idx = torch.randint(0, len(x1_data), (batch_size, ))
        batch = x1_data[idx]

        loss = train_step(model, optimizer, batch, device)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    plot_results(model, x1_data, device)
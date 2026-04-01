import math
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_moons
import torch
from torch import nn
from torch import optim
import cv2
import numpy as np

FRAMES_DIR = "frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

def generate_dataset(n_samples, noise, return_numpy=False, visualise_data=False):
    data, labels = make_moons(n_samples=n_samples, noise=noise)
    x1 = torch.tensor(data, dtype=torch.float32)
    x1 = (x1 - x1.mean(dim=0)) / x1.std(dim=0)

    if visualise_data:
        x_np = x1.numpy()
        plt.scatter(x_np[:, 0], x_np[:, 1], c=labels, cmap='coolwarm', s=5)
        plt.title("Moons Dataset (colored by class)")
        plt.show()

    if return_numpy:
        return x1.numpy()

    return x1


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        scaling_factor = math.log(10000) / (half_dim - 1)
        embeddings = torch.arange(half_dim, device=device) * -scaling_factor
        embeddings = torch.exp(embeddings)
        embeddings = embeddings[None, :] * t
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
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
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_mlp(t * 1000)
        x_input = torch.cat([x, t_emb], dim=1)
        return self.net(x_input)


def train_step(model, optimizer, x1, device):
    optimizer.zero_grad()
    x0 = torch.randn_like(x1)
    batch = x1.shape[0]
    t = torch.rand((batch, 1), device=device)
    t = t * (1 - 1e-3) + 1e-3
    xt = (1 - t) * x0 + t * x1
    target_velocity = x1 - x0
    v_pred = model(xt, t)
    loss = torch.mean((v_pred - target_velocity) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def sample(model, n_samples=1000, steps=100, device='cpu'):
    model.eval()
    x = torch.randn((n_samples, 2)).to(device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n_samples, 1), i / steps).to(device)
        v = model(x, t)
        x = x + v * dt
    model.train()
    return x.cpu().numpy()


def save_frame(model, x1_data, device, epoch, loss, loss_history):
    generated = sample(model, n_samples=2000, steps=100, device=device)
    real = x1_data[:2000].cpu().numpy()

    fig = plt.figure(figsize=(16, 5), facecolor='#0e0e14')

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # --- Plot 1: Overlay ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#0e0e14')
    ax1.scatter(real[:, 0], real[:, 1], color='#4fc3f7', alpha=0.25, s=4, label='Real')
    ax1.scatter(generated[:, 0], generated[:, 1], color='#ff6b6b', alpha=0.55, s=4, label='Generated')
    ax1.set_title("Real vs Generated", color='white', fontsize=11, pad=8)
    ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    ax1.tick_params(colors='#666')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333')

    # --- Plot 2: Velocity Field ---
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#0e0e14')
    grid = torch.linspace(-3, 3, 20)
    gx, gy = torch.meshgrid(grid, grid, indexing='ij')
    points = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(device)
    t_half = torch.full((points.shape[0], 1), 0.5).to(device)
    with torch.no_grad():
        v = model(points, t_half).cpu()
    magnitude = torch.sqrt(v[:, 0]**2 + v[:, 1]**2).numpy()
    magnitude = magnitude / (magnitude.max() + 1e-8)
    colors = plt.cm.plasma(magnitude)
    ax2.quiver(
        points[:, 0].cpu(), points[:, 1].cpu(),
        v[:, 0], v[:, 1],
        color=colors, alpha=0.85
    )
    ax2.set_title("Velocity Field (t=0.5)", color='white', fontsize=11, pad=8)
    ax2.tick_params(colors='#666')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333')

    # --- Plot 3: Loss Curve ---
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('#0e0e14')
    if len(loss_history) > 1:
        epochs_so_far = list(range(0, len(loss_history) * 100, 100))
        ax3.plot(epochs_so_far, loss_history, color='#a78bfa', linewidth=1.5)
        ax3.fill_between(epochs_so_far, loss_history, alpha=0.15, color='#a78bfa')
    ax3.set_title("Training Loss", color='white', fontsize=11, pad=8)
    ax3.set_xlabel("Epoch", color='#999', fontsize=9)
    ax3.set_ylabel("Loss", color='#999', fontsize=9)
    ax3.tick_params(colors='#666')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#333')

    fig.suptitle(
        f"Flow Matching  —  Epoch {epoch:,}  |  Loss {loss:.4f}",
        color='white', fontsize=13, fontweight='bold', y=1.02
    )

    path = os.path.join(FRAMES_DIR, f"frame_{epoch:06d}.png")
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#0e0e14')
    plt.close(fig)
    return path


def compile_video(frames_dir, output_path="flow_training.mp4", fps=10):
    frame_files = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith('.png')
    ])

    if not frame_files:
        print("No frames found.")
        return

    sample_frame = cv2.imread(frame_files[0])
    h, w = sample_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for f in frame_files:
        img = cv2.imread(f)
        writer.write(img)

    writer.release()
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Model initialized on: {device}")

    num_samples = 5000
    noise_ratio = 0.05
    batch_size = 256

    x1_data = generate_dataset(num_samples, noise_ratio).to(device)
    print(f"Moons dataset generated | Samples: {num_samples} | Noise: {noise_ratio}")
    print(f"Training... frames saved every 100 epochs to '{FRAMES_DIR}/'")

    loss_history = []

    for epoch in range(10001):
        idx = torch.randint(0, len(x1_data), (batch_size,))
        batch = x1_data[idx]
        loss = train_step(model, optimizer, batch, device)

        if epoch % 100 == 0:
            loss_history.append(loss)
            save_frame(model, x1_data, device, epoch, loss, loss_history)
            print(f"Epoch {epoch:>6} | Loss: {loss:.4f} | Frame saved")

    print("\nCompiling video...")
    compile_video(FRAMES_DIR, output_path="flow_training.mp4", fps=10)
    print("Done!")
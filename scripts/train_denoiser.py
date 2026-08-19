"""
Train a small 1D convolutional denoising autoencoder on seismic
waveforms from the 2023 Kahramanmaras sequence.

Since we don't have genuinely paired noisy/clean recordings, this
follows standard practice: cut real waveforms into overlapping windows,
synthetically add Gaussian noise to create (noisy, clean) training
pairs, and train the autoencoder to reconstruct the clean signal.

Usage:
    python3 scripts/train_denoiser.py
"""
import numpy as np
import obspy
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WAVEFORM_DIR = Path("data/waveforms")
OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_LEN = 512       # samples per training window
WINDOW_STRIDE = 128    # overlap between windows
NOISE_STD_RANGE = (0.1, 0.5)  # relative to signal std, randomized per window
EPOCHS = 30
BATCH_SIZE = 32


def load_and_window_traces():
    """Load all mseed files, normalize, cut into overlapping windows."""
    windows = []
    for f in sorted(WAVEFORM_DIR.glob("*.mseed")):
        st = obspy.read(str(f))
        for tr in st:
            data = tr.data.astype(np.float32)
            if len(data) < WINDOW_LEN:
                continue
            # z-score normalize per trace before windowing
            data = (data - data.mean()) / (data.std() + 1e-8)
            for start in range(0, len(data) - WINDOW_LEN, WINDOW_STRIDE):
                windows.append(data[start:start + WINDOW_LEN])
    windows = np.array(windows, dtype=np.float32)
    print(f"Built {len(windows)} training windows of length {WINDOW_LEN}")
    return windows


def add_noise(clean_batch):
    noise_std = np.random.uniform(*NOISE_STD_RANGE, size=(clean_batch.shape[0], 1)).astype(np.float32)
    noise = np.random.randn(*clean_batch.shape).astype(np.float32) * noise_std
    return (clean_batch + noise).astype(np.float32)


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=9, padding=4), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def snr_db(clean, denoised):
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - denoised) ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-12))


def train():
    clean_windows = load_and_window_traces()
    if len(clean_windows) < 20:
        raise RuntimeError(
            f"Only {len(clean_windows)} windows available - too few to train. "
            "Consider more stations or a longer waveform window in fetch_waveforms.py"
        )

    train_clean, test_clean = train_test_split(
        clean_windows, test_size=0.2, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_tensor = torch.tensor(train_clean).unsqueeze(1)  # (N, 1, L)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training on {device}, {len(train_clean)} windows, {EPOCHS} epochs...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for (clean_batch,) in loader:
            clean_np = clean_batch.squeeze(1).numpy()
            noisy_np = add_noise(clean_np)
            noisy_batch = torch.tensor(noisy_np).unsqueeze(1).to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()
            output = model(noisy_batch)
            loss = loss_fn(output, clean_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, loss: {total_loss/len(loader):.4f}")

    # Evaluate on held-out test windows
    model.eval()
    test_np = test_clean
    noisy_test = add_noise(test_np)
    with torch.no_grad():
        noisy_tensor = torch.tensor(noisy_test).unsqueeze(1).to(device)
        denoised = model(noisy_tensor).squeeze(1).cpu().numpy()

    snr_before = np.mean([snr_db(test_np[i], noisy_test[i]) for i in range(len(test_np))])
    snr_after = np.mean([snr_db(test_np[i], denoised[i]) for i in range(len(test_np))])
    print(f"\nMean SNR before denoising: {snr_before:.2f} dB")
    print(f"Mean SNR after denoising:  {snr_after:.2f} dB")
    print(f"Improvement: {snr_after - snr_before:.2f} dB")

    # Plot a few example before/after windows
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes):
        idx = i * (len(test_np) // 3)
        ax.plot(test_np[idx], label="Clean (original)", alpha=0.7, linewidth=1)
        ax.plot(noisy_test[idx], label="Noisy (input)", alpha=0.5, linewidth=0.8)
        ax.plot(denoised[idx], label="Denoised (model output)", alpha=0.9, linewidth=1)
        ax.set_ylabel("Amplitude (normalized)")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Sample")
    fig.suptitle(
        f"Seismic Waveform Denoising - Kahramanmaras Sequence\n"
        f"SNR improvement: {snr_after - snr_before:.2f} dB (mean over test set)"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "denoising_before_after.png", dpi=200)
    print(f"\nSaved figure to {OUT_DIR / 'denoising_before_after.png'}")


if __name__ == "__main__":
    train()
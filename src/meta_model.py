import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Meta model: simple MLP that maps state features -> next_loss
class MetaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def build_features_targets(df: pd.DataFrame):
    # Features at step t
    feats = df[['train_loss','train_acc','val_loss','val_acc','param_mean','param_std','param_l2','grad_l2','num_params']].values.astype(np.float32)
    # Target is next-step loss (val_loss as a stable estimate)
    next_val_loss = np.roll(df['val_loss'].values.astype(np.float32), -1)
    # Remove last step (no next target)
    feats = feats[:-1]
    next_val_loss = next_val_loss[:-1]
    return feats, next_val_loss

def train_meta(csv_path: str, epochs: int = 40, batch_size: int = 64, lr: float = 1e-3, hidden: int = 64, save_dir: str = "runs/meta"):
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    X, y = build_features_targets(df)

    # simple train/val split
    n = len(X)
    n_train = int(0.8 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    device = torch.device("cpu")
    model = MetaMLP(in_dim=X.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=batch_size, shuffle=False)

    history = {"epoch": [], "train_mse": [], "val_mse": []}
    for epoch in range(1, epochs+1):
        model.train()
        mse_sum, count = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            mse_sum += loss.item() * len(xb)
            count += len(xb)
        train_mse = mse_sum / max(1, count)

        model.eval()
        val_mse_sum, val_count = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device))
                loss = loss_fn(pred.cpu(), yb)
                val_mse_sum += loss.item() * len(xb)
                val_count += len(xb)
        val_mse = val_mse_sum / max(1, val_count)

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        print(f"[Meta][Epoch {epoch:03d}] train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(save_dir, "meta_history.csv")
    hist_df.to_csv(hist_csv, index=False)
    torch.save(model.state_dict(), os.path.join(save_dir, "meta_mlp.pt"))
    print(f"[Meta] Saved: {hist_csv} and meta_mlp.pt")

    # Save last predictions for analysis
    model.eval()
    with torch.no_grad():
        all_pred = model(torch.from_numpy(X)).numpy()
    np.savez(os.path.join(save_dir, "meta_predictions.npz"), y_true=y, y_pred=all_pred[n_train:])  # store val segment for analyze script

    return hist_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="runs/base/trajectory.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="runs/meta")
    args = parser.parse_args()

    train_meta(csv_path=args.csv, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, hidden=args.hidden, save_dir=args.save_dir)

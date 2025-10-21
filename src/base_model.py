import argparse
import os
import math
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# ---- Base classifier (small MLP) ----
class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def generate_dataset(n_samples=4000, noise=0.2, test_size=0.2, seed=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed)
    return (torch.from_numpy(X_train), torch.from_numpy(y_train)), (torch.from_numpy(X_val), torch.from_numpy(y_val))

def summarize_params_and_grads(model: nn.Module) -> Dict[str, float]:
    # Aggregate across all parameters
    with torch.no_grad():
        p_list = [p.view(-1) for p in model.parameters() if p.requires_grad]
        params = torch.cat(p_list) if p_list else torch.tensor([])

        # Gradient list (some grads can be None at beginning)
        g_list = []
        for p in model.parameters():
            if p.grad is not None:
                g_list.append(p.grad.view(-1))
        grads = torch.cat(g_list) if len(g_list) > 0 else torch.zeros(1)

        summary = {
            'param_mean': params.mean().item() if params.numel() > 0 else 0.0,
            'param_std': params.std().item() if params.numel() > 0 else 0.0,
            'param_l2': torch.linalg.vector_norm(params).item() if params.numel() > 0 else 0.0,
            'grad_l2': torch.linalg.vector_norm(grads).item() if grads.numel() > 0 else 0.0,
            'num_params': float(params.numel())
        }
        return summary

def accuracy(logits, y):
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()

def train_base(steps=300, batch_size=128, lr=0.03, hidden=64, seed=42, save_dir='runs/base'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    (X_train, y_train), (X_val, y_val) = generate_dataset()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512, shuffle=False)

    model = MLP(in_dim=2, hidden=hidden, out_dim=2)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    logs = []
    step_counter = 0
    loss_fn = nn.CrossEntropyLoss()

    train_iter = iter(train_loader)
    for step in range(steps):
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        model.train()
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()

        # Summarize with current gradients (before step)
        summary_before = summarize_params_and_grads(model)
        train_acc = accuracy(logits, yb)

        opt.step()

        # Evaluate quickly on val (optional: keep tiny for speed)
        model.eval()
        with torch.no_grad():
            v_losses, v_accs = [], []
            for xv, yv in val_loader:
                lv = loss_fn(model(xv), yv)
                v_losses.append(lv.item())
                v_accs.append(accuracy(model(xv), yv))
            val_loss = float(np.mean(v_losses))
            val_acc = float(np.mean(v_accs))

        log_record = {
            'step': step,
            'train_loss': loss.item(),
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            **summary_before
        }
        logs.append(log_record)

    # Save CSV
    import pandas as pd
    df = pd.DataFrame(logs)
    csv_path = os.path.join(save_dir, 'trajectory.csv')
    df.to_csv(csv_path, index=False)

    # Also save numpy for convenience
    np.savez(os.path.join(save_dir, 'trajectory_npz.npz'), **{k: np.array([r[k] for r in logs]) for k in logs[0].keys()})

    print(f"[Base] Saved trajectory with {len(logs)} steps to: {csv_path}")
    return csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="runs/base")
    args = parser.parse_args()

    train_base(steps=args.steps, batch_size=args.batch_size, lr=args.lr, hidden=args.hidden, seed=args.seed, save_dir=args.save_dir)

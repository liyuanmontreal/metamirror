import argparse
from base_model import train_base

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

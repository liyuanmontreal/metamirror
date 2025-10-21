import argparse
from meta_model import train_meta

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

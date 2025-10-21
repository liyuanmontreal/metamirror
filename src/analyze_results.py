import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(base_csv="runs/base/trajectory.csv", meta_npz="runs/meta/meta_predictions.npz", meta_hist="runs/meta/meta_history.csv"):
    df = pd.read_csv(base_csv)
    meta = np.load(meta_npz)
    y_true = meta["y_true"]     # validation y_true segment
    y_pred = meta["y_pred"]     # validation predictions (aligned)

    # Plot history (MSE)
    hist = pd.read_csv(meta_hist)
    plt.figure()
    plt.plot(hist["epoch"], hist["train_mse"], label="train_mse")
    plt.plot(hist["epoch"], hist["val_mse"], label="val_mse")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Meta Model Training Curve")
    plt.legend()
    os.makedirs("runs/figs", exist_ok=True)
    plt.savefig("runs/figs/meta_training_curve.png", dpi=160)
    plt.close()

    # Plot predictions vs ground truth (val segment)
    plt.figure()
    plt.plot(y_true, label="true_next_val_loss")
    plt.plot(y_pred, label="pred_next_val_loss")
    plt.xlabel("validation time index")
    plt.ylabel("next-step loss")
    plt.title("Meta Predictions vs Ground Truth (Validation)")
    plt.legend()
    plt.savefig("runs/figs/meta_pred_vs_true.png", dpi=160)
    plt.close()

    print("Saved figures to runs/figs/")

if __name__ == "__main__":
    main()

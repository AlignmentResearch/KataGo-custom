import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    DESCRIPTION = """Plots loss from PyTorch training."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "-dir", help="Path to the training run directory", type=Path, required=True
    )
    parser.add_argument(
        "-output", help="Path to write the loss plot", type=Path, required=True
    )
    args = parser.parse_args()

    train_metrics_file = args.dir / "train" / "t0" / "metrics_train.json"
    losses = []
    with open(train_metrics_file) as f:
        for line in f:
            metrics = json.loads(line)
            losses.append(metrics["loss"])

    plt.plot(losses)
    plt.ylabel("loss")
    plt.savefig(args.output)


if __name__ == "__main__":
    main()

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    DESCRIPTION = """Plots loss from PyTorch training."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "dirs", help="Path to the training run directory", type=Path, nargs="+"
    )
    parser.add_argument(
        "-output", help="Path to write the loss plot", type=Path, required=True
    )
    args = parser.parse_args()

    for d in args.dirs:
        train_metrics_file = d / "train" / "t0" / "metrics_train.json"
        nsamps = []
        losses = []
        with open(train_metrics_file) as f:
            for line in f:
                metrics = json.loads(line)
                nsamps.append(metrics["nsamp"])
                losses.append(metrics["loss"])
        plt.plot(nsamps, losses, label=d.name)

    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(args.output)


if __name__ == "__main__":
    main()

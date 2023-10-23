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
    parser.add_argument(
        "--validation",
        help="Print validation loss instead of train loss",
        action="store_true",
    )
    args = parser.parse_args()

    for d in args.dirs:
        file_modifier = "val" if args.validation else "train"
        metrics_file = d / "train" / "t0" / f"metrics_{file_modifier}.json"
        nsamp_key = "nsamp_train" if args.validation else "nsamp"
        nsamps = []
        losses = []
        with open(metrics_file) as f:
            for line in f:
                metrics = json.loads(line)
                nsamps.append(metrics[nsamp_key])
                losses.append(metrics["loss"])
        plt.plot(nsamps, losses, label=d.name)

    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(args.output)


if __name__ == "__main__":
    main()

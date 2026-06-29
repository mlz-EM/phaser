import csv
from pathlib import Path

import matplotlib.pyplot as plt
import tifffile


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "grad_trial0301_benchmark_results.csv"
OUT_PATH = ROOT / "grad_trial0301_benchmark_plot.png"


def main() -> None:
    rows = []
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["sigma"] = int(row["sigma"])
            row["weight"] = float(row["weight"])
            row["avg_iter_seconds"] = float(row["avg_iter_seconds"])
            rows.append(row)

    rows.sort(key=lambda row: row["sigma"], reverse=True)

    develop = [row for row in rows if row["code"] == "develop"]
    new = [row for row in rows if row["code"] == "pr52"]

    x = range(len(develop))
    labels = []
    for row in develop:
        if row["sigma"] == 0:
            labels.append("s=0\nno layers")
        else:
            labels.append(f"s={row['sigma']}\nw={row['weight']}")

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(3, len(develop), height_ratios=[1.2, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(list(x), [row["avg_iter_seconds"] for row in develop], marker="o", label="develop")
    ax.plot(list(x), [row["avg_iter_seconds"] for row in new], marker="o", label="new (pr52)")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Avg iter time, last 150 iters (s)")
    ax.set_xlabel("Sigma / weight")
    ax.set_title("grad_trial0301 benchmark")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    image_axes = []

    for col, row in enumerate(develop):
        img = tifffile.imread(row["image_path"])
        ax_img = fig.add_subplot(gs[1, col])
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title(f"develop sigma={row['sigma']}")
        ax_img.axis("off")
        image_axes.append(ax_img)

    for col, row in enumerate(new):
        img = tifffile.imread(row["image_path"])
        ax_img = fig.add_subplot(gs[2, col], sharex=image_axes[col], sharey=image_axes[col])
        ax_img.imshow(img, cmap="gray")
        ax_img.set_title(f"new sigma={row['sigma']}")
        ax_img.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)


if __name__ == "__main__":
    main()

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEVELOP_CSV = ROOT / "develop_layer_blur_benchmark.csv"
PR52_CSV = ROOT / "pr52_layer_blur_benchmark.csv"
OUT_PATH = ROOT / "layer_blur_benchmark_plot.png"


def load_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "sigma": float(row["sigma"]),
                "avg_seconds": float(row["avg_seconds"]),
            })
    rows.sort(key=lambda row: row["sigma"], reverse=True)
    return rows


def main() -> None:
    develop = load_rows(DEVELOP_CSV)
    pr52 = load_rows(PR52_CSV)

    x = range(len(develop))
    labels = [f"s={int(row['sigma'])}" for row in develop]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(list(x), [row["avg_seconds"] for row in develop], marker="o", label="develop")
    ax.plot(list(x), [row["avg_seconds"] for row in pr52], marker="o", label="pr-52")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Average blur time (s)")
    ax.set_xlabel("Sigma")
    ax.set_title("Layer Blur Benchmark")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)


if __name__ == "__main__":
    main()

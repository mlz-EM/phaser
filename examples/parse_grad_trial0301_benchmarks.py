import csv
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "bench_logs"
OUT_CSV = ROOT / "grad_trial0301_benchmark_results.csv"

LINE_RE = re.compile(
    r"Finished iter\s+(?P<iter>\d+)\s*/\s*200\s+\[(?P<minutes>\d+):(?P<seconds>\d{2}\.\d{3})\]"
)
NAME_RE = re.compile(
    r"(?P<code>main|develop|pr52)_sigma(?P<sigma>\d+)(?:_w(?P<weight>[0-9]+p[0-9]+))?\.log$"
)


def parse_time_to_seconds(minutes: str, seconds: str) -> float:
    return int(minutes) * 60.0 + float(seconds)


def main() -> None:
    rows = []
    for path in sorted(LOG_DIR.glob("*.log")):
        match = NAME_RE.match(path.name)
        if match is None:
            continue

        times = []
        for line in path.read_text(errors="replace").splitlines():
            m = LINE_RE.search(line)
            if m is None:
                continue
            times.append(parse_time_to_seconds(m.group("minutes"), m.group("seconds")))

        if len(times) != 200:
            raise RuntimeError(f"{path.name}: expected 200 iteration timings, found {len(times)}")

        tail = times[-150:]
        sigma = int(match.group("sigma"))
        weight = 0.0 if sigma == 0 else float(match.group("weight").replace("p", "."))
        label = {
            "main": "old",
            "develop": "develop",
            "pr52": "new",
        }[match.group("code")]
        out_dir = ROOT / f"bench_{label}_sigma{sigma}" if sigma == 0 else ROOT / f"bench_{label}_sigma{sigma}_w{str(weight).replace('.', 'p')}"

        rows.append({
            "code": match.group("code"),
            "label": label,
            "sigma": sigma,
            "weight": weight,
            "avg_iter_seconds": mean(tail),
            "image_path": str(out_dir / "object_phase_sum_iter200.tiff"),
            "log_path": str(path),
        })

    rows.sort(key=lambda row: (row["sigma"], row["code"]))

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["code", "label", "sigma", "weight", "avg_iter_seconds", "image_path", "log_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

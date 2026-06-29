import argparse
import csv
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, help="Path to iter200.h5")
    parser.add_argument("--sigmas", nargs="+", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out", required=True, help="CSV output path")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from phaser.state import ReconsState
    from phaser.hooks.regularization import RegularizeLayersProps
    from phaser.engines.common.regularizers import RegularizeLayers

    state_path = Path(args.state)
    template = ReconsState.read_hdf5(state_path).to_xp(jnp)

    rows: list[dict[str, float | str | int]] = []

    for sigma in args.sigmas:
        sim = template.copy()
        reg = RegularizeLayers(None, RegularizeLayersProps(sigma=sigma, weight=1.0))

        reg.init_state(sim)

        sim, _ = reg.apply_iter(sim, None)
        jax.block_until_ready(sim.object.data)

        t0 = time.perf_counter()
        for _ in range(args.repeats):
            sim, _ = reg.apply_iter(sim, None)
        jax.block_until_ready(sim.object.data)
        elapsed = time.perf_counter() - t0

        rows.append({
            "sigma": sigma,
            "repeats": args.repeats,
            "avg_seconds": elapsed / args.repeats,
            "state_path": str(state_path),
        })

    out_path = Path(args.out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma", "repeats", "avg_seconds", "state_path"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

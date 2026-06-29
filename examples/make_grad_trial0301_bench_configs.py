import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "grad_trial0301.json"

SIGMA_WEIGHT_PAIRS = [
    (200, 0.1),
    (150, 0.2),
    (100, 0.3),
    (50, 0.5),
]

CODE_LABELS = [
    ("main", "old"),
    ("develop", "develop"),
    ("pr52", "new"),
]


def main() -> None:
    base = json.loads(BASE_PATH.read_text())

    for code_key, code_label in CODE_LABELS:
        for sigma, weight in SIGMA_WEIGHT_PAIRS:
            data = json.loads(json.dumps(base))
            tag = f"{code_label}_sigma{sigma}_w{str(weight).replace('.', 'p')}"
            data["name"] = f"grad_trial0301_{tag}"

            engine = data["engines"][0]
            engine["save_options"]["out_dir"] = f"bench_{tag}"
            engine["solvers"]["object"]["eps"] = 0.00000001
            engine["solvers"]["probe"]["eps"] = 0.00000001

            layers_constraint = engine["iter_constraints"][3]
            layers_constraint["sigma"] = float(sigma)
            layers_constraint["weight"] = float(weight)

            out_path = ROOT / f"grad_trial0301_{code_key}_sigma{sigma}_w{str(weight).replace('.', 'p')}.json"
            text = json.dumps(data, indent=4) + "\n"
            text = text.replace("1e-08", "0.00000001")
            out_path.write_text(text)

        data = json.loads(json.dumps(base))
        tag = f"{code_label}_sigma0"
        data["name"] = f"grad_trial0301_{tag}"

        engine = data["engines"][0]
        engine["save_options"]["out_dir"] = f"bench_{tag}"
        engine["solvers"]["object"]["eps"] = 0.00000001
        engine["solvers"]["probe"]["eps"] = 0.00000001
        engine["iter_constraints"] = [
            constraint
            for constraint in engine["iter_constraints"]
            if constraint.get("type") != "layers"
        ]

        out_path = ROOT / f"grad_trial0301_{code_key}_sigma0.json"
        text = json.dumps(data, indent=4) + "\n"
        text = text.replace("1e-08", "0.00000001")
        out_path.write_text(text)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for exp, run in [
        ("2D", ROOT / "two_step_2d/runs/kpp2d/full_2d_v2_dpfix2/metrics.json"),
        ("3D", ROOT / "two_step_3d/runs/kpp3d/full_3d_otgeo_dpfix2/metrics.json"),
    ]:
        data = json.loads(run.read_text())
        print(exp)
        for row in data["levels"]:
            print(
                f"  -log2s={row['log2sigma']:.2f} "
                f"cold={row.get('w2_cold', float('nan')):.4f} "
                f"mf={row['w2_mf']:.4f} dp={row['w2_dp']:.4f}"
            )


if __name__ == "__main__":
    main()

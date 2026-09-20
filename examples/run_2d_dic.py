"""Run single-camera planar DIC over an image sequence (no calibration needed).

Usage:
    python3 examples/run_2d_dic.py images/ --pattern "*.tif" \\
        --subset-radius 15 --grid-step 10 --out results/
"""
import argparse
import os

from pydic.io_utils import load_sequence, save_fields_csv
from pydic.pipeline import Dic2D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir")
    ap.add_argument("--pattern", default="*.tif")
    ap.add_argument("--subset-radius", type=int, default=15)
    ap.add_argument("--grid-step", type=int, default=10)
    ap.add_argument("--out", default="dic_results")
    args = ap.parse_args()

    paths = load_sequence(args.images_dir, args.pattern)
    print(f"{len(paths)} frames found")
    if len(paths) < 2:
        raise SystemExit("need at least a reference frame + one deformed frame")

    dic = Dic2D(subset_radius=args.subset_radius, grid_step=args.grid_step)
    dic.set_reference(paths[0])
    print(f"{len(dic.points)} measurement points on the grid")

    results = dic.run_sequence(paths[1:])

    os.makedirs(args.out, exist_ok=True)
    for t, res in enumerate(results, start=1):
        strain = dic.compute_strain(res)
        save_fields_csv(
            os.path.join(args.out, f"frame_{t:04d}.csv"),
            res.points,
            {
                "u": res.u, "v": res.v, "zncc": res.zncc, "valid": res.valid.astype(int),
                "exx": strain.exx, "eyy": strain.eyy, "exy": strain.exy,
            },
        )
        print(f"frame {t}: {res.valid.mean()*100:.1f}% points converged")

    print(f"wrote per-frame CSVs to {args.out}/")


if __name__ == "__main__":
    main()

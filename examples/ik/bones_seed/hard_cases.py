"""Pick a set of hard clips for tuning, from evaluate.py's per clip metrics
(--save): for each failure mode, frax's worst clips, and those where it trails
SEED's own G1 retargets the most. Mirrored takes (_M) count as one clip, and few
clips are picked per category, to cover many kinds of motion

Writes one clip per line (with why it was picked), worst first per failure mode,
for retarget.py, evaluate.py, and view.py's --clip_list
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from dataset import DATA_DIR, read_metadata
from evaluate import load_metrics
from seed_g1 import SEED_FPS

OUT = Path(__file__).parent / "hard_cases.txt"
MIN_SECONDS = 2.0  # Shorter clips aren't ranked by snap rate
# Failure mode: how bad a clip is for it, from its frax and nvidia metrics
CRITERIA = {
    "worst_snap": lambda f, n: f["worst_snap"],
    "snaps_per_min": lambda f, n: (
        f["snaps"] / f["frames"] * SEED_FPS * 60
        if f["frames"] > MIN_SECONDS * SEED_FPS
        else 0.0
    ),
    "jerks_per_min": lambda f, n: (
        f["jerks"] / f["frames"] * SEED_FPS * 60
        if f["frames"] > MIN_SECONDS * SEED_FPS
        else 0.0
    ),
    "off_target": lambda f, n: f["off_target"],
    "rot_err": lambda f, n: f["rot_err"],
    "skate": lambda f, n: np.nan_to_num(f["skate"]),
    "floor_depth": lambda f, n: f["floor_depth"],
    "self_coll_depth": lambda f, n: f["self_coll_depth"],
    # Where frax does worse than SEED's G1
    "rot_err_vs_nvidia": lambda f, n: f["rot_err"] - n["rot_err"],
    "off_target_vs_nvidia": lambda f, n: f["off_target"] - n["off_target"],
    "skate_vs_nvidia": lambda f, n: np.nan_to_num(f["skate"] - n["skate"]),
}


def main(args):
    meta = {r["filename"]: r for r in read_metadata(args.data_dir)}
    results = load_metrics(Path(args.metrics))
    frax, nvidia = results["frax"], results["nvidia"]
    picked = {}  # clip: [(criterion, value)]
    for criterion, badness in CRITERIA.items():
        scores = {c: badness(m, nvidia[c]) for c, m in frax.items()}
        takes, categories, n = set(), Counter(), 0
        for clip in sorted(scores, key=lambda c: -scores[c]):
            take = clip.removesuffix("_M")
            category = meta[clip]["category"]
            if scores[clip] <= 0 or n == args.per_criterion:
                break
            if take in takes or categories[category] == args.per_category:
                continue
            takes.add(take)
            categories[category] += 1
            n += 1
            picked.setdefault(clip, []).append((criterion, scores[clip]))

    lines = [
        f"# {len(picked)} hard clips, from {args.metrics} ({len(frax)} clips)",
        "# clip  criteria=value (frax's metric, or its excess over nvidia's)"
        + "  package/category",
    ]
    for clip, reasons in picked.items():
        why = " ".join(f"{c}={v:.2f}" for c, v in reasons)
        lines.append(f"{clip}  {why}  {meta[clip]['package']}/{meta[clip]['category']}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"{len(picked)} clips to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", help="evaluate.py --save's per clip metrics (CSV)")
    parser.add_argument("--data_dir", default=DATA_DIR, help="SEED dataset root")
    parser.add_argument("--out", default=OUT, help="Clip list to write")
    parser.add_argument(
        "--per_criterion", type=int, default=20, help="Clips per failure mode"
    )
    parser.add_argument(
        "--per_category", type=int, default=3, help="Most clips per category, per mode"
    )
    main(parser.parse_args())

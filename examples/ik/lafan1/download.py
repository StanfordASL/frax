"""Download the LAFAN1 dataset (Ubisoft La Forge) into data/lafan1,
for retarget.py and evaluate.py

LAFAN1 is released under CC BY-NC-ND 4.0, for non-commercial use only (see
LICENSE_URL). Skipped if the BVH files are already there
"""

import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from lafan1_g1 import DATA_DIR

REPO_URL = "https://github.com/ubisoft/ubisoft-laforge-animation-dataset"
ZIP_URL = f"{REPO_URL}/raw/master/lafan1/lafan1.zip"
LICENSE_URL = f"{REPO_URL}/blob/master/license.txt"


def download(url: str, path: Path):
    """Download url to path, printing progress"""

    def progress(blocks, block_size, total):
        done = min(blocks * block_size, total)
        print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, path, reporthook=progress)
    print()


def main(args):
    data_dir = Path(args.data_dir)
    existing = sorted(data_dir.glob("*.bvh"))
    if existing:
        print(f"{len(existing)} BVH files already in {data_dir}: nothing to do")
        return
    print(
        "LAFAN1 (Ubisoft La Forge) is licensed under CC BY-NC-ND 4.0: "
        + f"non-commercial use only. See {LICENSE_URL}"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir.parent) as tmp:
        zip_path = Path(tmp) / "lafan1.zip"
        print(f"Downloading {ZIP_URL}")
        download(ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            # Only the BVH files, flattened: by base name, so no member can
            # write outside data_dir
            members = [
                m
                for m in archive.infolist()
                if m.filename.endswith(".bvh")
                and not m.is_dir()
                and not Path(m.filename).name.startswith(".")
            ]
            for member in members:
                target = data_dir / Path(member.filename).name
                target.write_bytes(archive.read(member))
    print(f"Extracted {len(members)} BVH files to {data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default=DATA_DIR, help="Where to put the BVH files"
    )
    main(parser.parse_args())

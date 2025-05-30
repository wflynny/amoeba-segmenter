#!/usr/bin/env python
import os
import re
import argparse
from pathlib import Path
from subprocess import run

ROOT_DIR = os.environ.get("AMOEBASEGMENTER_ROOT_DIR", None)
if not ROOT_DIR:
    raise Exception("Need to source root config file!")
ROOT_DIR = Path(ROOT_DIR)

TRAINING_DATA_DIR = ROOT_DIR / "data" / "training"
VALIDATION_DATA_DIR = ROOT_DIR / "data" / "validation"
PRODUCTION_DATA_DIR = ROOT_DIR / "data" / "production"

VALIDATION_RESULTS_DIR = VALIDATION_DATA_DIR / "results"
PRODUCTION_RESULTS_DIR = PRODUCTION_DATA_DIR / "results"

if not VALIDATION_RESULTS_DIR.exists():
    VALIDATION_RESULTS_DIR.mkdir(exist_ok=True)

if not VALIDATION_RESULTS_DIR.exists():
    PRODUCTION_RESULTS_DIR.mkdir(exist_ok=True)


def get_int_from_match(pattern, string, idx=-1):
    result = 0
    match = pattern.match(string)
    if match:
        result = int(match.groups()[-1])
    return result


def get_max_series(root_image):
    suffix_pattern = re.compile(".*_(\d+).ome.tiff")

    converted_images = list(
        sorted(root_image.parent.glob(f"{root_image.name}*.ome.tiff"))
    )

    converted_max_series = max(
        [
            get_int_from_match(suffix_pattern, str(convert_path.name))
            for convert_path in converted_images
        ]
    )
    return converted_max_series


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    parser.add_argument("--skip-to", default=None)
    args = parser.parse_args()
    expt_dir = Path(args.experiment_dir)

    if not expt_dir.exists():
        print("Path [expt_dir] does not exist!")
        exit()
    elif expt_dir.is_file():
        print("Path [expt_dir] is not a directory!")
        exit()

    expt_name = expt_dir.name
    results_dir = PRODUCTION_RESULTS_DIR / expt_name
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)

    sorted_raw_image_files = list(sorted(expt_dir.glob("*.nd2")))
    print(f"Found {len(sorted_raw_image_files)} raw images to process.")

    if args.skip_to:
        idx = [f.name for f in sorted_raw_image_files].index(args.skip_to)
        sorted_raw_image_files = sorted_raw_image_files[idx:]
        print(f"Skipping first {idx} files...")

    for image_file in sorted_raw_image_files:
        max_series_int = get_max_series(image_file)

        run(
            [
                "sbatch",
                f"--array=0-{max_series_int}",
                f"{ROOT_DIR}/scripts/21_process.sbatch",
                image_file,
                results_dir,
            ]
        )

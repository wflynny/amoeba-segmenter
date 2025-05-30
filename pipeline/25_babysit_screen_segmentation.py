#!/usr/bin/env python
### SLURM HEADER
# SBATCH --job-name=babysit-segmentation
# SBATCH --output=microscopy_babysitter.log
# SBATCH --mail-type=FAIL
# SBATCH --mail-user=bill.flynn@jax.org

# SBATCH --account=singlecell
# SBATCH --partition=batch
# SBATCH --qos=normal
# SBATCH --time=48:00:00
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=1
# SBATCH --mem=4GB

# SBATCH --export=ALL
### SLURM HEADER

import os
import re
import time
from pathlib import Path
from subprocess import run, PIPE

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


def get_queue_length():
    proc = run("squeue -r -u $USER | wc -l", shell=True, stdout=PIPE)
    return int(proc.stdout.decode("ascii").strip())


if __name__ == "__main__":
    expt_dir = PRODUCTION_DATA_DIR / "microscopy"

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
    print(f"Found {len(sorted_raw_image_files)} raw images to process.", flush=True)

    MAX_QUEUE_LENGTH = 2000
    idx = 0
    while idx < len(sorted_raw_image_files):
        image_file = sorted_raw_image_files[idx]
        max_series_int = get_max_series(image_file)
        if get_queue_length() < MAX_QUEUE_LENGTH:
            print(
                f"[{time.ctime()}]: Submitting image {image_file}: {max_series_int} jobs",
                flush=True,
            )
            run(
                [
                    "sbatch",
                    f"--array=0-{max_series_int}",
                    f"{ROOT_DIR}/pipeline/21_process.sbatch",
                    image_file,
                    results_dir,
                ]
            )
            idx += 1
        time.sleep(2)
    print(f"[{time.ctime()}]: No more images to process. Exiting", flush=True)

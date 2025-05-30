#!/usr/bin/env python
import re
import argparse
from pathlib import Path
from subprocess import run, PIPE


series_count_pattern = re.compile("Series count = (\d+)")
suffix_pattern = re.compile(".*_(\d+).ome.tiff")
series_grep_cmd = (
    "singularity run -B /sc "
    "${AMOEBASEGMENTER_ROOT_DIR}/containers/img/bftools-6.13.0.sif "
    "showinf -nopix '{}' | grep 'Series count'"
)

def get_int_from_match(pattern, string, idx=-1):
    result = 0
    match = pattern.match(string)
    if match:
        result = int(match.groups()[-1])
    return result


def verify_image_file(root_image: Path):
    # assume all conversions append "_\d+.ome.tiff" to root_image name
    # run bioformats showinf + grep to get series count
    # collect conversion tiffs
    # assert same number
    # assert highest number = nseries + 1
    proc = run(
        series_grep_cmd.format(str(root_image.resolve())), 
        check=True, stdout=PIPE, shell=True
    )
    grep_output = proc.stdout.decode("ascii").strip()
    series_count = get_int_from_match(series_count_pattern, grep_output)

    converted_images = list(sorted(
        root_image.parent.glob(f"{root_image.name}*.ome.tiff")
    ))
    converted_max_series = max([
        get_int_from_match(suffix_pattern, str(convert_path.name))
        for convert_path in converted_images
    ])

    all_good = series_count == len(converted_images) == converted_max_series + 1
    if all_good:
        print(f"PASS: {root_image}")
    else:
        print(f"FAIL: {root_image}: {series_count} {len(converted_images)} {converted_max_series+1}")


def verify_dir(indir: Path, skip_to: str = None, ext: str = ".nd2"):
    # find all nd2 files
    # run verify_image_file()

    image_files = list(sorted(indir.glob(f"*{ext}")))
    skip_complete = skip_to is None
    for image in image_files:
        if not skip_complete:
            if image.name != skip_to: 
                print("Skipping ", image)
                continue
            else:
                skip_complete = True
        verify_image_file(image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    parser.add_argument("-s", "--skip-to")
    args = parser.parse_args()

    indir = Path(args.indir)
    assert indir.exists()

    verify_dir(indir, skip_to=args.skip_to)

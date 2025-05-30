#!/usr/bin/env python
import warnings
warnings.simplefilter("ignore")
import re
import argparse
import pandas as pd
from pathlib import Path


PROPS_PATTERN = re.compile("([0-9\._A-z- ]+).nd2_(\d+).ome.tiff.props.csv")


def collate_properties_files(expt_dir, expt_name, ext=".props.csv"):
    found_files = list(sorted(expt_dir.glob(f"*{ext}")))
    print(f"{str(expt_dir)}: Found {len(found_files)} {ext} files.")
    baby_dfs = []
    for prop_file in found_files:
        pat_match = PROPS_PATTERN.match(prop_file.name)
        if not pat_match:
            continue

        plate_name, series = pat_match.groups()
        df = pd.read_csv(prop_file, index_col=0, header=0)
        if len(df) == 0:
            print("I'm here", expt_name, plate_name, series)
            df.loc[0, "label"] = 1
            df = df.infer_objects(copy=False).fillna(0)

        df["experiment"] = expt_name
        df["plate"] = plate_name
        df["series"] = int(series) + 1
        baby_dfs.append(df)

    mega_df = pd.concat(baby_dfs, ignore_index=True, axis=0)
    return mega_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir")
    args = parser.parse_args()

    expt_dir = Path(args.experiment_dir)
    if not expt_dir.exists():
        raise Exception(f"Directory passed doesn't exist: [{expt_dir}]")

    expt_name = expt_dir.stem
    out_name = expt_dir / f"{expt_name}.results.csv.gz"

    results = collate_properties_files(expt_dir, expt_name)

    results.to_csv(out_name)
    print(f"Results saved to {out_name}")

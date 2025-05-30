#!/usr/bin/env python
import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = os.environ.get("AMOEBASEGMENTER_ROOT_DIR", None)
if not ROOT_DIR:
    raise Exception("Need to source root config file!")
ROOT_DIR = Path(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))
del sys

from cyst_segmenter.platemap import CANONICAL_WELL_ORDERING, serpentine

## need a function that accounts for all the things below:
##  specific wells: \w\d{1,2}(?:\s(ONLY|ONLY RETAKE|ONLY RETAKE AGAIN))
##  whole/partial rows: A1-12; D11-1  \w\d+-\d+(?:[\s_]+(read|retake))(?:\s\d)
##  whole row sets:  A-D; E-H; A-B
##  full plate retake

# need this because the diff-media_exp1 naming is bonkers
EMERGENCY_MAPPING = {
    "Plate 1_C1-11": "Plate 1_COLUMN1-11_0",
    "Plate 1_C12": "Plate 1_COLUMN12_0",
    "Plate 1_C12001": "Plate 1_COLUMN12_1",
    "Plate 1_C12_RA-D": "Plate 1_COLUMN12_2",
    "Plate 2_A1": "Plate 2_A1_0",
    "Plate 2_A1001": "Plate 2_A1_1",
    "Plate 2_A1002": "Plate 2_A1_2",
    "Plate 2_A1003": "Plate 2_A1_3",
    "Plate 2_C1-11": "Plate 2_COLUMN1-11_0",
    "Plate 2_C12": "Plate 2_COLUMN12_0",
    "Plate 2_C12001": "Plate 2_COLUMN12_1",
    "Plate 2_C12002": "Plate 2_COLUMN12_2",
    "Plate 2_C12003": "Plate 2_COLUMN12_3",
}


def wells_from_row(row, start=1, end=12):
    if start > end:
        start, end = end, start

    step = -1 if row in "BDFH" else 1
    return [f"{row}{k}" for k in range(start, end + 1)][::step]


def sanitize_plate_modifiers(mod):
    """takes in a modifer and returns 1 or more well names"""
    match_well1 = re.match(
        "^(?:read2_)?(\w\d{1,2})"
        "(?:[\s_](ONLY RETAKE AGAIN|ONLY RETAKE|ONLY|newarea|newarea_lowercondenser))?$",
        mod,
    )
    match_well2 = re.match("^(\w\d{1,2})(?:\d+)?$", mod)
    match_well3 = re.match("^(\w\d)(?:_\d+)?$", mod)
    match_col1 = re.match("^COLUMN(\d)-(\d+)_\d+$", mod)
    match_col2 = re.match("^COLUMN(\d+)_\d+$", mod)
    match_row1 = re.match("^(\w)(\d{1,2})-(\d{1,2})(?:_read\s?\d?)?$", mod)
    match_row2 = re.match("^row\s*(\w)$", mod)
    match_rows = re.match("^(\w)-(\w)(?:\s*(?:RETAKE AGAIN|RETAKE|_read\s*\d))?$", mod)
    match_plate = re.match("^(full plate retake|[_\s]?read\s?\d[_\s]?|)$", mod)
    rows = "ABCDEFGH"

    if match_well1:
        return [match_well1.groups()[0]]

    if match_well2:
        return [match_well2.groups()[0]]

    if match_well3:
        return [match_well3.groups()[0]]

    if match_col1:
        c1, c2 = match_col1.groups()
        _ = list(serpentine(8, int(c2)+1 - int(c1)).keys())
        #_.remove("B11")
        return _

    if match_col2:
        col = match_col2.groups()[0]
        return [f"{c}{col}" for c in "ABCDEFGH"]

    if match_row1:
        row, start, end = match_row1.groups()
        return wells_from_row(row, int(start), int(end))

    if match_row2:
        row = match_row2.groups()[0]
        return wells_from_row(row)

    if match_rows:
        r1, r2 = match_rows.groups()
        return sum(
            [wells_from_row(r) for r in rows[rows.index(r1) : rows.index(r2) + 1]], []
        )

    if match_plate:
        return CANONICAL_WELL_ORDERING


def reconcile(objects, global_metadata, outfile):
    all_series = (
        objects.groupby(["experiment", "plate", "series", "simple_plate", "plate_mod"])
        .size()
        .reset_index()
        .iloc[:, :-1]
    )

    series_data = all_series.copy()
    series_data["experiment_harmonized"] = series_data.experiment
    series_data = series_data[
        series_data.experiment_harmonized.isin(
            global_metadata.experiment_harmonized.unique()
        )
    ]
    series_data = series_data.set_index(
        ["experiment_harmonized", "simple_plate", "series"]
    )
    modified_series_data = series_data[series_data.plate_mod != ""].copy()

    modified_series_data["corrected_well"] = None
    modified_series_data["corrected_series"] = None
    modified_series_data = (
        modified_series_data.reset_index()
        .set_index(["experiment_harmonized", "simple_plate", "plate_mod"])
        .sort_index()
    )

    for (expt, p, pm), df in modified_series_data.groupby(
        ["experiment", "simple_plate", "plate_mod"]
    ):
        # this gets a list of wells corresponding contained in the modified image file
        mod_wells = sanitize_plate_modifiers(pm)
        assert mod_wells, (pm, mod_wells)

        # based on the full_metadata, what are the wells we expect to find
        # the wells returned from the modified plate name should be a superset of those in the metadata
        # this is because A1-12 or rowA may only contain 10-11 images based on which wells are empty or contain controls.
        expectation = global_metadata[
            global_metadata.experiment_harmonized.isin([expt])
            & global_metadata.simple_plate.isin([p])
        ]
        expected_wells = expectation.well[expectation.well.isin(mod_wells)].tolist()
        assert set(expected_wells) - set(mod_wells) == set(), (
            expected_wells,
            mod_wells,
        )
        # it's important we keep this order
        mod_well_subset = [w for w in mod_wells if w in expected_wells]

        # create a map of well -> series
        correct_well_to_series_map = dict(
            zip(
                *expectation.loc[
                    expectation.well.isin(mod_well_subset), ["well", "series"]
                ].values.T
            )
        )
        # use the map above to know to which actual series the reimaged wells correspond
        # e.g. a plate of H11-1 will have images 1, 2, ..., 11 but should be mapped to 86, 87, ..., 96
        correct_series = [correct_well_to_series_map[w] for w in mod_well_subset]
        # for some ungodly reason, we have some images that contain fewer wells than expected
        if len(df) < len(correct_series):
            correct_series = correct_series[:len(df)]
            mod_well_subset = mod_well_subset[:len(df)]

        modified_series_data.loc[(expt, p, pm), "corrected_well"] = mod_well_subset
        modified_series_data.loc[(expt, p, pm), "corrected_series"] = correct_series

    replacements = modified_series_data.reset_index()[
        [
            "experiment_harmonized",
            "simple_plate",
            "corrected_series",
            "corrected_well",
            "plate",
            "series",
        ]
    ].drop_duplicates(
        subset=["experiment_harmonized", "simple_plate", "corrected_series"],
        keep="last",
        ignore_index=True,
    )
    replacements.columns = [
        "experiment_harmonized",
        "simple_plate",
        "original_series",
        "well",
        "replace_plate",
        "replace_series",
    ]
    replacements.to_csv(outfile)
    return replacements


def prepare_data(args):
    full_metadata = pd.read_csv(args.platemaps, index_col=0)
    full_metadata = full_metadata[~full_metadata.experiment_harmonized.isnull()]

    all_objects = pd.concat(
        [
            pd.read_csv(f, index_col=0)
            for f in sorted(Path(args.data_root).rglob(f"*/*{args.data_ext}"))
        ],
        ignore_index=True,
    )

    # yup, nothing to see here, totally normal
    all_objects.plate = all_objects.plate.map(EMERGENCY_MAPPING).fillna(all_objects.plate)
    
    plate_extras = all_objects.plate.str.extract(".*[pP]late.*?(\d+)[A-Z]?[_ ]?(.*)")
    all_objects[["simple_plate", "plate_mod"]] = plate_extras
    all_objects.simple_plate = all_objects.simple_plate.astype(int)

    if args.reconcile:
        replacements = reconcile(all_objects, full_metadata, args.reconcile_output)
    else:
        replacements = pd.read_csv(args.reconcile_output, index_col=0)

    data_mod = all_objects[all_objects.plate_mod != ""].copy()
    data_final = (
        all_objects[all_objects.plate_mod == ""]
        .drop("plate_mod", axis=1)
        .set_index(["experiment", "simple_plate", "series"])
        .sort_index()
        .copy()
    )

    idx_cols = ["experiment", "simple_plate", "series"]
    common_cols = data_final.columns.intersection(data_mod.columns).tolist()

    new_datums = []
    for _, row in replacements.iterrows():
        idx = (row.experiment_harmonized, row.simple_plate, row.original_series)
        try:
            data_final.drop(idx, axis=0, inplace=True)
            print("removed ", idx)
        except KeyError:
            print("error, cant remove", idx)
            pass

        datum = data_mod.loc[
            data_mod.experiment.isin([row.experiment_harmonized])
            & data_mod.plate.isin([row.replace_plate])
            & data_mod.series.isin([row.replace_series]),
            common_cols + idx_cols,
        ]
        datum["series"] = row.original_series
        datum = datum.set_index(idx_cols)
        new_datums.append(datum)

    final_object_data = pd.concat([data_final] + new_datums)

    print("final object data created")
    final_object_data.to_parquet(args.object_data)


def quantify_single(object_store, metadata, dataset, output_root, version, rt, gt):
    all_objects = pd.read_parquet(args.object_data, engine="pyarrow")

    # aparent mpp = 0.325um / pixel.  Cyst diameter should be >10um
    # also get rid of elongated shapes, likely debris or cysts we can't easily distinguish from neighbors
    filtered_objects = all_objects[
        (all_objects.equivalent_diameter_area * 0.325 > 10)
        & (all_objects.eccentricity < 0.85)
    ].copy()
    filtered_objects["is_red"] = (filtered_objects.intensity_mean_red > rt).astype(int)
    filtered_objects["is_green"] = (filtered_objects.intensity_mean_green > gt).astype(
        int
    )
    filtered_objects["is_red-green"] = (
        filtered_objects.is_red & filtered_objects.is_green
    ).astype(int)
    filtered_objects["is_red-or-green"] = (
        filtered_objects.is_red | filtered_objects.is_green
    ).astype(int)
    filtered_objects["is_red-xor-green"] = (
        filtered_objects.is_red ^ filtered_objects.is_green
    ).astype(int)

    aggs = dict(
        cysts=pd.NamedAgg(column="label", aggfunc=lambda z: len(np.unique(z))),
        red_cysts=pd.NamedAgg(column="is_red", aggfunc="sum"),
        green_cysts=pd.NamedAgg(column="is_green", aggfunc="sum"),
        red_green_cysts=pd.NamedAgg(column="is_red-green", aggfunc="sum"),
        red_or_green_cysts=pd.NamedAgg(column="is_red-or-green", aggfunc="sum"),
        red_xor_green_cysts=pd.NamedAgg(column="is_red-xor-green", aggfunc="sum"),
    )
    per_well_measurements = (
        filtered_objects.groupby(["experiment", "simple_plate", "series"])
        .agg(**aggs)
        .reset_index()
    )

    per_well_measurements["red_frac"] = (
        per_well_measurements.red_cysts / per_well_measurements.cysts
    )
    per_well_measurements["green_frac"] = (
        per_well_measurements.green_cysts / per_well_measurements.cysts
    )
    per_well_measurements["red-green_frac"] = (
        per_well_measurements.red_green_cysts / per_well_measurements.cysts
    )

    per_well_measurements["percent_red"] = per_well_measurements["red_frac"] * 100
    per_well_measurements["percent_green"] = per_well_measurements["green_frac"] * 100
    per_well_measurements["percent_red-green"] = (
        per_well_measurements["red-green_frac"] * 100
    )

    per_well_measurements["percent_red-is-green"] = (
        per_well_measurements.red_green_cysts / per_well_measurements.red_cysts * 100
    )
    per_well_measurements["percent_red-or-green"] = (
        per_well_measurements.red_or_green_cysts / per_well_measurements.cysts * 100
    )
    per_well_measurements["percent_survival"] = (
        100 - per_well_measurements["percent_red"]
    )

    expt_names = metadata.loc[
        metadata.dataset == dataset, "experiment_harmonized"
    ].unique()

    merged = []
    for (expt, plate), df in per_well_measurements[
        per_well_measurements.experiment.isin(expt_names)
    ].groupby(["experiment", "simple_plate"]):
        meta = metadata.loc[
            (metadata.experiment_harmonized.isin([expt]))
            & (metadata.simple_plate == plate),
            [
                "series",
                "condition",
                "condition_name",
                "condition_quantity",
                "condition_quantity_numeric",
                "condition_quantity_unit",
                "condition_time",
            ],
        ]
        _merged = pd.merge(df, meta, left_on="series", right_on="series")
        merged.append(_merged)
    final = pd.concat(merged, ignore_index=True).sort_values(
        ["condition_name", "condition_quantity_numeric", "condition_time"]
    )
    final.to_csv(f"{output_root}/{dataset}_{version}.csv")


def quantify_data(args):  # , red_threshold=2500, green_threshold=12000):
    full_metadata = pd.read_csv(args.platemaps, index_col=0)
    full_metadata = full_metadata[~full_metadata.experiment_harmonized.isnull()]

    if args.dataset is None:
        datasets = full_metadata.dataset.unique()
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        print("writing output for ", dataset)

        quantify_single(
            args.object_data,
            full_metadata,
            dataset,
            args.data_output,
            args.version,
            args.red_threshold,
            args.green_threshold,
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    subparsers = cli.add_subparsers(dest="command")
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--object-data",
        default=ROOT_DIR / "data" / "production" / "all_object_data.parquet",
    )
    parent_parser.add_argument(
        "--data-root", default=ROOT_DIR / "data" / "production" / "results"
    )
    parent_parser.add_argument(
        "--data-output", default=ROOT_DIR / "data" / "production" / "results-combined"
    )
    parent_parser.add_argument(
        "--platemaps", default=ROOT_DIR / "data" / "plate_maps" / "all_platemaps.csv"
    )

    prep_parser = subparsers.add_parser(
        "prepare", parents=[parent_parser], help="reconcile reimaging"
    )
    prep_parser.add_argument("--reconcile", action="store_true", default=False)
    prep_parser.add_argument(
        "--reconcile-output",
        default=ROOT_DIR / "data" / "plate_maps" / "reimaging_correspondence.csv",
    )
    prep_parser.add_argument("--data-ext", default=".csv.gz")
    prep_parser.set_defaults(run_command=prepare_data)

    quant_parser = subparsers.add_parser(
        "quantify", parents=[parent_parser], help="generate results per condition"
    )
    quant_parser.add_argument("-d", "--dataset", default=None)
    quant_parser.add_argument("-r", "--red-threshold", type=int, default=2500)
    quant_parser.add_argument("-g", "--green-threshold", type=int, default=12500)
    quant_parser.add_argument("-v", "--version")
    quant_parser.set_defaults(run_command=quantify_data)
    args = cli.parse_args()

    args.run_command(args)

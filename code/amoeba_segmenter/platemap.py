"""
The utility functions in this file are not necessary for segmentation.
However, they are useful for manipulating the experimental metadata 
files that have been used to track experimental conditions in microwell
plates.

Moreover, there is (likely now unneeded) functionality to (attempt) to
harmonize many of the strange image names, experiment names, and other
small naming inconsistencies present during the data collection.
"""
import re
import json
import pandas as pd
from string import ascii_uppercase


def serpentine(nrows=8, ncols=12):
    rows = list(ascii_uppercase[:nrows])
    cols = [str(k) for k in range(1, ncols + 1)]
    wells = []
    for i, r in enumerate(rows):
        for c in cols[:: 1 - 2 * (i % 2)]:
            wells.append(r + c)
    return dict(zip(wells, range(1, len(wells) + 1)))


CANONICAL_WELL_ORDERING = list(serpentine().keys())
PLATEMAP_FORMAT = re.compile("(.*)_(Plate [A-z0-9-]+).*?\.csv")


def map_wells_to_series_with_gaps(well_list):
    """
    Assuming a row-oriented serpentine imaging pattern starting at A1,
    assign a 1-indexed series ID to each well.

    Given a list of wells for one 96wp, it must be len(well_list) <= 96
    the way we've written the platemap parser, all "empty"/null wells
    have been excluded, so that the wells in the well_list we're given
    all correspond to a material condition.
    """
    well_list = list(well_list)
    k = 1
    series = {}
    for well in CANONICAL_WELL_ORDERING:
        if well in well_list:
            series[well] = k
            k += 1
    return series


def read_raw_platemap(infile, na_values=[]):
    na_vals = ["", "empty", "unused"] + na_values
    na_vals += list(map(str.capitalize, na_vals)) + list(map(str.upper, na_vals))

    m = pd.read_csv(infile, index_col=0, na_values=na_vals)
    m.columns.name = "col"
    m.index.name = "row"
    return m


def load_experiment_maps(map_file):
    """
    Mapping from "Detailed.Experiment Name #1" -> "detailed_expt1"
    stored as list of {"source": "Long Name", "dest": "short_name", "group": "dataset_name"}
    """
    expt_mapping = {}
    dataset_mapping = {}
    with open(map_file, "r") as fin:
        for item in json.load(fin):
            expt_mapping[item["source"]] = item["dest"]
            dataset_mapping[item["source"]] = item["group"]
    return expt_mapping, dataset_mapping


def platemap_to_longform(map_file, na_values=[]):
    platemap = read_raw_platemap(map_file, na_values=na_values)
    long = pd.melt(platemap.reset_index(), value_name="condition", id_vars=["row"])
    long["well"] = long.row + long.col.astype(str)
    long = long.dropna()

    long["series"] = long["well"].map(map_wells_to_series_with_gaps(long.well))

    match = PLATEMAP_FORMAT.match(map_file.name)
    expt, plate = match.groups()
    long["experiment"] = expt
    long["plate"] = plate
    long["simple_plate"] = long.plate.str.extract(".*(\d+).*.?")

    long["condition_time"] = long.condition.str.extract("\(([0-9\.]+ hrs)\)").fillna("")
    long[
        ["condition_quantity", "condition_quantity_numeric", "condition_quantity_unit"]
    ] = long.condition.str.extract("((^[0-9\.]+)\s?(%|µM|μM)?)").fillna("")
    long["condition_quantity"] = long.condition.str.extract(
        "(^[0-9\.]+%?\s?(?:µM)?)"
    ).fillna("")
    long["condition_name"] = long.apply(
        lambda r: r.condition[
            len(r.condition_quantity) : len(r.condition)
            - len(r.condition_time)
            - 2 * (len(r.condition_time) > 0)
        ].strip(" "),
        axis=1,
    )

    return long.reset_index()[
        [
            "experiment",
            "plate",
            "simple_plate",
            "row",
            "col",
            "well",
            "series",
            "condition",
            "condition_name",
            "condition_quantity",
            "condition_quantity_numeric",
            "condition_quantity_unit",
            "condition_time",
        ]
    ]


def consolidate_all_maps(
    longform_map_list, experiment_mapping=None, dataset_mapping=None
):
    all_maps = pd.concat(longform_map_list, ignore_index=True)
    if experiment_mapping is not None:
        all_maps["experiment_harmonized"] = all_maps.experiment.map(experiment_mapping)
    if dataset_mapping is not None:
        all_maps["dataset"] = all_maps.experiment.map(dataset_mapping)
    return all_maps

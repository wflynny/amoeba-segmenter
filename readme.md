# Segmentation Pipeline for Acanthamoeba image-based drug screens

This repository supports the work of C. Flynn, R. Colon-Rios, A. Harmez, B.
Kazmierczak in the Microbial Pathogenesis department at Yale University.

## Description

This repository is one of a small related collection of repositories which use
this tool to segment images of amoebas in their various lifeforms and use the
resulting segmentation to quantify lifeform viability via IF staining.

This repository only holds the segmentation code and helper scripts to segment
large numbers of Nikon .nd2 files on an HPC system with a SLURM-based job
scheduler.  Other related repositories store the data and results for the
various applications of this tool.

Related repositories:
- [[cyst-viability-assay](https://github.com/wflynny/cyst-viability-assay)] -
  Validation of this tool and its use to measure A. castellanii Cyst viability
  via live/dead imaging by Ethidium Homodimer-1 (EthD-1) and Calcein-AM
  staining.
- [[amoebicidal-small-molecules]()] - Work in Progress

## Installation

Install the necessary packages using conda, mamba, or their ilk via:
```{bash}
conda create -n amoeba-segment -f env.yml
```

From there, you can either install locally via pip:
```{bash}
cd amoeba-segmenter
pip install -e .
```

Or just add it to your path since there are no real intrapackage dependencies
```{bash}
here=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
export AMOEBASEGMENTER_ROOT_DIR="${here}"
export AMOEBASEGMENTER_CODE_DIR="${AMOEBASEGMENTER_ROOT_DIR}/code"
```
and use in python scripts with:
```{python}
import os
import sys
from pathlib import Path

CODE_DIR = os.environ.get("AMOEBASEGMENTER_CODE_DIR", None)
if not CODE_DIR:
    raise Exception("Need to source root config file!")
CODE_DIR = Path(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
del sys

from amoeba_segmenter.segmenter import SimpleAmoebaSegmenter
```

To facilitate the above, you can source the provided `config.bash` file which
will define these environment variables.


## Usage

### Standalone usage or in a notebook

See example utilization in `notebooks/validate_segmentation.ipynb`.  Briefly,
using the installation methods above, you should be able to:

```
from amoeba_segmenter.segmenter import SimpleAmoebaSegmenter

seg = SimpleAmoebaSegmenter("/path/to/image.ome.tif", debug=True)
seg.squeeze_image(projection_method="max")
seg.segment(diameter=30)
seg.show_segmentation()
```

To avoid repeating the segmentation call, segmentation results can be saved (as
.npy files) and loaded back in later:
```
seg.save_segmentation(".")

# provide sample input path
seg2 = SimpleAmoebaSegmenter("/path/to/image.ome.tif", debug=True)
seg2.squeeze_image(projection_method="max")
seg2.load_previous_segmentation(".")
seg2.show_segmentation()
```

### Headless usage
To run on a large set of images, follow the steps below:

#### Convert all image stacks to single series TIFFs
Create a singularity/apptainer image of bftools.  An example definition file is
in `containers/def/bftools-6.13.0.def` but you can likely just pull a
representative image from dockerhub or elsewhere.  

Alternatively, if you have
access to bftools locally (which you can find
[here](https://bio-formats.readthedocs.io/en/latest/users/comlinetools/)), then
you should export the environment variable
`BFTOOLS_PATH=/path/to/bftools/bfconvert` which will override subsequent calls
to apptainer.

From there, data is assumed to be in the following structure:
```{bash}
data/
|- production/
|-- experiment-name1/
|--- expt1-plate1.nd2
|--- expt1-plate2.nd2
|--- ...
```
If it's not in that structure, just create a symlink (assuming the data is
stored on the same storage array):
```{bash}
cd data/production
ln -s /path/to/data/expt .
```

Use the `experiment-name` alone to convert nd2 files to tiff.
```{bash}
pipeline/10_convert_nd2_to_tif.sh experiment-name
```
This will search for all .nd2 files under `data/production/experiment-name` and
convert each to an OME.TIF using the `pipeline/11_bfconvert_proxy` script,
which uses a SLURM job scheduler to create individual jobs for each .nd2 file
found.

Once complete, you can attempt to use the optional script to verify the
conversion completed successfully:
```{bash}
python pipeline/15_verify_conversion.py experiment-name
```
which will compare the number of tifs generated with the series number reported
by bftools/showinf.

#### Segment tiffs
Running
```{bash}
python pipeline/20_launch_production.py experiment-name
```
will submit SLURM jobs of `pipeline/21_process.sbatch` which in turn calls
`pipeline/22_process.py` on each file individually. All segmentation results
will be stored under
```{bash}
data/
|- production/
|-- experiment-name1/
|--- expt1-plate1.nd2
|--- expt1-plate2.nd2
|--- ...
|-- results/
|--- experiment-name1/
|---- expt1-plate1-s0.npy
|---- ...
```

In the case that you have thousands of individual tiffs,
`20_launch_production.py` is greedy and will attempt to queue processing of all
images at once. This generally makes job schedulers unhappy, so depending on
your computing system, you may want to modify it to use a job array of fixed
size or use the optional active monitoring script:
```{bash}
python pipeline/25_babysit_screen_segmentation.py experiment-name
```
which will limit the number of processing and queued jobs to some
`MAX_QUEUE_LENGTH`, currently set to 2000.

#### Quantification of signal within segmented objects
This requires a rewrite, as it currently relies on on-the-fly parsing of file
names to associate specific series IDs with wells in a plate, but this is a
terrible idea for many reasons.

Essentially current logic is to first run:
```{bash}
python pipeline/30_aggregate_experiment.py experiment-name
```
This simply takes the segmented object properties from each tiff in an
experiment and collects them into a big dataframe for uniform quantification of
all objects in an experiment.  This is an easier route than quantification then
aggregation because quantification may need to be adjusted on a per-experiment
basis to account for different object sizes (trophs vs cysts), batch effects
like background illumination problems, contamination, etc.

From there quantification is done with:
```{bash}
python pipeline/40_quantify.py --help
```

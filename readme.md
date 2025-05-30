# Segmentation Pipeline for Acanthamoeba image-based drug screens

This repository supports the work of C. Flynn, R. Colon Rios, A. Harmez, B.
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

Install using conda, mamba, or their ilk via:
```{bash}
conda create -n amoeba-segment -f env.yml
```

## Usage

### Standalone usage or in a notebook


### Headless usage

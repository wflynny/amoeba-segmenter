from pathlib import Path
ROOT_DIR = (Path(__file__).parent / "..").resolve()
import sys
sys.path.insert(0, str(ROOT_DIR))
del sys

from amoeba_segmenter.segmenter import SimpleAmoebaSegmenter


TRAINING_DATA_DIR = ROOT_DIR / "data" / "training"
VALIDATION_DATA_DIR = ROOT_DIR / "data" / "validation"
VALIDATION_RESULTS_DIR = VALIDATION_DATA_DIR / "results"
if not VALIDATION_RESULTS_DIR.exists():
    VALIDATION_RESULTS_DIR.mkdir(exist_ok=True)


def process(fname):
    print(f"Segmenting image {fname}")
    seg = SimpleAmoebaSegmenter(fname, debug=False)
    seg.squeeze_image(projection_method="max")
    seg.segment(diameter=30)
    print(f"Segmentation for image {fname} complete")
    seg.save_segmentation(VALIDATION_RESULTS_DIR)


if __name__ == "__main__":
    validation_dir = VALIDATION_DATA_DIR / "Exp.8_2022.10.16-20"
    sorted_files = list(sorted(validation_dir.glob("*.ome.tiff")))
    for image_file in sorted_files:
        process(image_file)

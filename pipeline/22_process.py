import os
import sys
import argparse
from pathlib import Path

CODE_DIR = os.environ.get("AMOEBASEGMENTER_CODE_DIR", None)
if not CODE_DIR:
    raise Exception("Need to source root config file!")
CODE_DIR = Path(CODE_DIR)
sys.path.insert(0, str(CODE_DIR))
del sys

from amoeba_segmenter.segmenter import SimpleAmoebaSegmenter


def process(fname, res_dir, model_name):
    print(f"Segmenting image {fname}")
    seg = SimpleAmoebaSegmenter(fname, debug=False)
    seg.squeeze_image(projection_method="max")
    seg.segment(model_name=model_name, diameter=30)
    print(f"Segmentation for image {fname} complete")
    print(f"Saving results to {res_dir}")
    seg.save_segmentation(res_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file")
    parser.add_argument("result_dir")
    parser.add_argument("--model", default="cyto2")
    args = parser.parse_args()

    process(Path(args.image_file), Path(args.result_dir), args.model)

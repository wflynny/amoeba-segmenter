import warnings
import numpy as np
import pandas as pd
import cellpose.models

import tifffile as tf

from skimage.color import label2rgb
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops_table

CAN_PLOT = True
try:
    import matplotlib.pyplot as plt
    import cmocean as cmo

    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["figure.facecolor"] = "white"
except ImportError:
    CAN_PLOT = False



def rescale(im):
    return im / im.max()


class SimpleAmoebaSegmenter:
    """
    This class will take one image and attempt to segment cysts or trophs, assuming the following:

    - each file will be a tiff file containing one series image
    - that series image will be of the form:
        (T, Z, C, Y, X, S) = (1, z, 3, y, x, 1)
        where
            C1 = phase contrast brightfield
            C2 = red (EthD-1 / dead)
            C3 = green (Calcein AM / viable)
    - each image will need to projected down to a single z-plane
        by default, this is just a max projection, though others like sum/avg are available by
        design

    - after projection over z, we need to squeeze the image to guarantee we get a (c, y, x) image

    - the BF image will be segmented, then the signal from the red channel will be overlayed and
      thresholded within the BF segmented mask.

    Notes:
    For parallelization purposes, we expect multi series images (e.g. an .nd2 file of a whole
    plate) to be converted to individual tiff files for each series.  See the (`scripts`)
    directory for an example script.

    The simplest way to convert multi series images with `bioformats/bfconvert` will retain
    the complete, original OME metadata in each of the split files. The Tifffile package TiffFile
    reader will attempt similar behavior as the bioformats importer in FIJI, such that it will
    read multiple files to satisfy the image description in the metadata.  Even if we pass the
    `_multifile=False` kwarg, TiffFile will then not read the metadata correctly and separate out
    planes from channels.  The solution is to ensure that bfconvert uses the `-series` flag to
    separate out each series appropriately.

    We choose to use TiffFile vs something like skimage.io.imread because the latter seems to
    clobber over certain features of tifffiles, and I don't want to take any chances here.
    """

    BF = 0
    RED = 1
    GREEN = 2

    PROPERTIES = [
        "label",
        "intensity_mean",
        "intensity_min",
        "intensity_max",
        "area",
        "area_filled",
        "area_convex",
        "num_pixels",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "equivalent_diameter_area",
    ]

    def __init__(self, image_path, use_gpu: bool = False, debug: bool = False):
        self.debug = debug
        self.image_path = image_path

        self.load_models(use_gpu)
        self.load_image()

    def load_models(self, use_gpu):
        self.nuclei_model = cellpose.models.Cellpose(gpu=use_gpu, model_type="nuclei")
        self.cyto2_model = cellpose.models.Cellpose(gpu=use_gpu, model_type="cyto2")
        self.cyto3_model = cellpose.models.Cellpose(gpu=use_gpu, model_type="cyto3")
        self.cpsam_model = cellpose.models.Cellpose(gpu=use_gpu, model_type="cpsam")

    def log(self, msg):
        if self.debug:
            print(msg)

    def load_image(self):
        raw_tiff = tf.TiffFile(self.image_path, is_imagej=False, is_mmstack=False)
        raw_tiff_series = raw_tiff.series

        # we make a hard assumption that each file only has one series
        if len(raw_tiff_series) != 1:
            raise Exception(f"File [{self.image_path}] contains more than 1 series")
        # this gives us an image that is TZCYXS
        self.raw_image = raw_tiff_series[0].asarray(squeeze=False)

        t, z, c, y, x, s = self.raw_image.shape
        self.n_zplanes = z
        self.image_width = x
        self.image_height = y
        self.n_channels = c
        self.mask = None

    def squeeze_image(self, projection_method="max"):
        """
        The nd2 images used to develop this class will typically only contain CYX.
        However, when loaded with TiffFile, extra dimenions for series, time, etc.
        will be added of unit length. This attempts to provide a standard interface
        to take a (1,Z,3,Y,X,1) image down to (3,Y,X) image by squeezing the dimensions
        of unit size and projecting with an arbitrary aggregation function over Z.
        """
        try:
            projection_func = getattr(np, projection_method)
        except AttributeError:
            raise (
                f"Invalid projection method: {projection_method}.  Not found in numpy"
            )
        self.projection_method = projection_method
        # raw image (tzcyxs) -- projection over z --> (tcyxs) -- squeeze --> cyx
        self.image = np.squeeze(projection_func(self.raw_image, axis=1))
        self.log(f"Image loaded. (c,y,x) = {self.image.shape}")

    def normalize(self, im):
        """
        Many of the training and testing images had nonuniform background intensity
        or overall illumination.  This attempts to fix that.
        """
        if self.projection_method == "max":
            return equalize_adapthist(im)
        return im

    def _mask_image(self, mask, channel, return_background=False):
        masked = self.image[channel].copy()
        masked -= masked.min()
        if return_background:
            masked[mask > 0] = 0
        else:
            masked[mask <= 0] = 0
        return masked

    def make_color_masks(self):
        self.masked_red = self._mask_image(self.mask, self.RED, return_background=False)
        self.masked_green = self._mask_image(
            self.mask, self.GREEN, return_background=False
        )
        self.bg_red = self._mask_image(self.mask, self.RED, return_background=True)
        self.bg_green = self._mask_image(self.mask, self.GREEN, return_background=True)
        self.log("Color channels masked.")

    def generate_prop_table(self):
        rn_keys = [p for p in self.PROPERTIES if p.startswith("intensity")]
        red_props = pd.DataFrame(
            regionprops_table(
                self.mask, intensity_image=self.masked_red, properties=self.PROPERTIES
            )
        ).rename({k: f"{k}_red" for k in rn_keys}, axis="columns")

        green_props = pd.DataFrame(
            regionprops_table(
                self.mask, intensity_image=self.masked_green, properties=self.PROPERTIES
            )
        ).rename({k: f"{k}_green" for k in rn_keys}, axis="columns")

        self.prop_table = red_props.merge(green_props)
        # ensure prop table is not empty
        if len(self.prop_table) == 0:
            self.prop_table.loc[0, "label"] = 1
            self.prop_table = self.prop_table.infer_objects(copy=False).fillna(0)

        self.prop_table["mean_bg_red"] = (
            self.image[self.RED].mean() - self.image[self.RED].min()
        )
        self.prop_table["median_bg_red"] = (
            np.median(self.image[self.RED]) - self.image[self.RED].min()
        )
        self.prop_table["mean_bg_green"] = (
            self.image[self.GREEN].mean() - self.image[self.GREEN].min()
        )
        self.prop_table["median_bg_green"] = (
            np.median(self.image[self.GREEN]) - self.image[self.GREEN].min()
        )
        self.log("Object properties quantified.")

    def segment(self, model_name="cyto2", diameter=None):
        """
        Load the CellPose model specified. Original development was done with cyto2
        but now cyto3 and cytosam are available and should likely perform better.
        Segment the brightfield image and use resulting segmentation mask to
        quantify color channels (red & green).
        """
        self.log("Starting segmentation.")
        if diameter is None:
            diameter = self.image_width // 100
        self.log(f"Using diameter {diameter}.")

        to_segment = np.dstack(
            (
                rescale(self.normalize(self.image[self.BF])),
                np.zeros_like(self.image[self.BF]),
            )
        )

        model = getattr(self, model_name)
        self.mask, *garbage = model.eval(
            to_segment,
            diameter=diameter,
            channels=[0, 0],
            invert=False,
            flow_threshold=0.4,
            do_3D=False,
            normalize=False,
        )
        self.log("BF segmentation complete.")

        self.make_color_masks()
        self.generate_prop_table()

        self.log("All processing complete.")
        self.outputs = np.dstack(
            (self.mask, self.masked_red, self.masked_green, self.bg_red, self.bg_green)
        )

    def save_segmentation(self, parent_dir, prefix=None):
        """
        Save segmentation mask as an .npy file and the segmentation properties table
        as a .csv
        """
        if prefix is None:
            prefix = f"{parent_dir}/{self.image_path.name}"
        np.save(prefix + ".segmentation.npy", self.outputs)
        self.prop_table.to_csv(prefix + ".props.csv")

    def load_previous_segmentation(self, parent_dir, prefix=None):
        if prefix is None:
            prefix = f"{parent_dir}/{self.image_path.name}"
        obj = np.load(prefix + ".segmentation.npy")

        self.mask = obj[..., 0]
        self.make_color_masks()
        self.prop_table = pd.read_csv(prefix + ".props.csv", index_col=0)
        self.outputs = np.dstack(
            (self.mask, self.masked_red, self.masked_green, self.bg_red, self.bg_green)
        )

    def recalculate_props(self):
        # is this needed anymore?
        # only really useful during debugging prop table generation or adding more
        # properties
        if self.mask is None:
            self.log("Masks are not loaded. Cannot regenerate props.")
        self.generate_prop_table()

    def show_segmentation(self):
        if not CAN_PLOT:
            warnings.warn("Warning: plotting functionality not available")
            return

        fig, axs = plt.subplots(
            3, 4, dpi=200, figsize=(10, 7)
        )  # , sharex=True, sharey=True)

        axs[0, 0].set_title("Bright Field")
        axs[0, 0].imshow(self.image[self.BF], cmap="binary")
        axs[1, 0].imshow(self.mask > 0, cmap="binary")
        axs[2, 0].imshow(label2rgb(self.mask, bg_color=(255, 255, 255)))

        axs[0, 1].set_title("Red signal")
        axs[0, 1].imshow(self.image[self.RED], cmap="cmo.amp")
        axs[1, 1].imshow(self.masked_red, cmap="cmo.thermal")
        axs[2, 1].imshow(self.bg_red, cmap="cmo.thermal")
        axs[1, 1].set_title("Within segmentation")
        axs[2, 1].set_title("Outside segmentation")

        axs[0, 2].set_title("Green signal")
        axs[0, 2].imshow(self.image[self.GREEN], cmap="cmo.tempo")
        axs[1, 2].imshow(self.masked_green, cmap="cmo.thermal")
        axs[2, 2].imshow(self.bg_green, cmap="cmo.thermal")
        axs[1, 2].set_title("Within segmentation")
        axs[2, 2].set_title("Outside segmentation")

        histparams = dict(bins=np.arange(0, 2**16, 100), histtype="step")
        axs[0, 3].hist(
            self.bg_red.ravel(), color="0.5", label="Background", **histparams
        )
        axs[0, 3].hist(
            self.masked_red.ravel(), color="r", label="In Cysts", **histparams
        )
        axs[0, 3].legend(loc="upper right", frameon=False, fontsize="x-small")
        axs[1, 3].hist(
            self.bg_green.ravel(), color="0.5", label="Background", **histparams
        )
        axs[1, 3].hist(
            self.masked_green.ravel(), color="g", label="In Cysts", **histparams
        )
        axs[1, 3].legend(loc="upper right", frameon=False, fontsize="x-small")

        histparams = dict(bins=np.arange(0, 2**15, 50), histtype="step")
        axs[2, 3].hist(self.prop_table.intensity_mean_red, color="r", **histparams)
        axs[2, 3].hist(self.prop_table.intensity_mean_green, color="g", **histparams)

        for ax in axs[:, 3].flat:
            ax.set_yscale("log")

        for ax in axs[:, :3].flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()

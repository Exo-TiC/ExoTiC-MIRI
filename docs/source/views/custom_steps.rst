Custom step descriptions
========================

Here we list each of the available custom steps, and include a
description of the algorithms and arguments.

Drop integrations
-----------------

This stage 1 step enables the user to drop integrations from a data
chunk, most likely because these groups are too severely affected by
systematics to be worth processing. This step may also be useful
if the user wants to test pipelines on only a small subset of data.

**Parameters**

    | input_model : jwst.datamodels.RampModel
    | The input JWST RampModel.

    | drop_integrations : list of integers
    | This list represents the integrations that will be dropped from
      the data chunk. The integers are zero-indexed.

**Returns**

    | output_model : jwst.datamodels.RampModel
    | A RampModel with updated integrations, unless the step is skipped in
      which case the input_model is returned.

Drop groups
-----------

This steps enables the user to drop groups from each integration
which may be adversely affecting the ramps.

**Parameters**

    | input_model : jwst.datamodels.RampModel
    | The input JWST RampModel.

    | drop_groups : list of integers
    | This list represents the groups that will be dropped from
      each integration. The integers are zero-indexed.

**Returns**

    | output_model : jwst.datamodels.RampModel
    | A RampModel with updated groups, unless the step is skipped in
      which case the input_model is returned.

Regroup
-------

This steps enables the user to regroup integrations, comprised
of n groups, into several smaller integrations, comprised of m
groups, where n is a multiple of m.

**Parameters**

    | input_model : jwst.datamodels.RampModel
    | The input JWST RampModel.

    | n_groups : integer
    | The new number of groups per integration.

**Returns**

    | output_model : jwst.datamodels.RampModel
    | A RampModel with updated integration groupings, unless the step
      is skipped in which case the input_model is returned.

Reference pixel
---------------

This steps enables the user to apply corrections to their group-
level images, using the reference pixels available to the MIRI LRS
subarray. Reference pixels are used to correct for the non-ideal
behaviour of the four amplifiers, which may add values to pixels
dependent on amplifier, row, and group.

The simplest form of the algorithm computes the median reference pixel
correction for each of the four amplifiers, and subtracts it from
the corresponding pixels in the rest of the subarray. Further options
enable this correction to be separated by odd and even rows, or to be
computed as a running median up each column.

**Parameters**

    | input_model : jwst.datamodels.RampModel
    | The input JWST RampModel.

    | smoothing_length : integer
    | Median smooth reference pixel values over this pixel length in the
      column direction per amplifier.

    | odd_even_rows : boolean
    | Option to treat odd and even rows separately.

**Returns**

    | output_model : jwst.datamodels.RampModel
    | A RampModel with the reference pixel correction applied, unless
      the step is skipped in which case `input_model` is returned.


Extract 1d
----------

A step that will sky subtract, trace find, and extract 1d spectra
from the 2d rate images. This step assumes the photom step has not been
run, or at least input data units are DN/s.

The background/sky is estimated within specified column bounds. Within
this region the background can be computed as a constant values, via a
sigma clipped mean, or as a series of polynomials fits to each row.
These polynomial estimates can be optionally smoothed in the column
direction.

The spectral trace location (in x) is found by fitting a Gaussian to
the stacked rows, and this defines the centre of the extraction window
for each integration.

The spectra are extracted using one of several available algorithms: box
extraction using a fixed-width top-hat aperture, optimal extraction
following Horne (1986), and several other experimental algorithms are
also implemented.

**Parameters**

    | input_model : jwst.datamodels.CubeModel
    | The input JWST CubeModel.

    | bkg_algo : string ("constant", "polynomial")
    | The algorithm for background subtraction.

    | bkg_region : list of integers
    | Defines the region from which the background is estimated. The
      integers correspond to subarray columns (start, stop, start, stop) for
      coverage either side of the spectral trace.

    | bkg_poly_order : integer
    | The order of the polynomial fitted to each row for background estimation.

    | bkg_smoothing_length : integer
    | Median smooth background values over this pixel length in the column
      direction.

    | extract_algo : string ("box", "optimal", "anchor")
    | The algorithm for spectral extraction.

    | extract_region_width : integer
    | The full width of the extraction region, centred on the spectral trace.

    | extract_poly_order : integer
    | The order of the polynomials used in optimal extraction.

    | max_iter : integer
    | Maximum iterations in the spectral extraction. Only used if
      extract_algo="anchor".

**Returns**

    | output_model : jwst.datamodels.MultiSpecModel
    | A MultiSpecModel containing extracted 1d spectra. The spectra for
      each integration are packaged as a list of pandas.DataFrames in
      MultiSpecModel.spectra. If the step is skipped the `input_model`
      is returned.

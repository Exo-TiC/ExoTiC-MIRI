Stage 2 steps
=============

The stage 2 steps perform mode-specific corrections on the rate-images,
followed by spectral extraction. Below we detail the workings of several
steps that you may find useful for improving the quality of your reduction
beyond the default
`JWST stage 2 <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_spec2.html>`_.
For a complete list of ExoTiC-MIRI steps and their workings, see the
:doc:`API <api/api>`.

The following examples assume you have setup the pipelines and loaded
in a _rateints.fits data segment. For example:

.. code-block:: python

    import os

    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    os.environ["CRDS_PATH"] = "path/to/your/local/crds_dir"
    os.environ["CRDS_CONTEXT"] = "jwst_1100.pmap"

    from jwst import datamodels
    from jwst.pipeline import calwebb_spec2

    # Load data segment.
    proc = datamodels.CubeModel("path/to/your/data/jwxyz-segnnn_mirimage_rateints.fits")


Get the wavelength map
----------------------

To get the MIRI LRS wavelength map you can run
:class:`GetWavelengthMap <exotic_miri.reference.GetWavelengthMap>`. Note
that you must run jwst.calwebb_spec2.assign_wcs_step and jwst.calwebb_spec2.srctype_step
before this step.

.. code-block:: python

    from exotic_miri.reference import GetWavelengthMap


    custom_get_wavelength_map = GetWavelengthMap()
    stsci_assign_wcs = calwebb_spec2.assign_wcs_step.AssignWcsStep()
    stsci_srctype = calwebb_spec2.srctype_step.SourceTypeStep()

    proc = stsci_srctype.call(proc)
    sproc = stsci_assign_wcs.call(proc)
    wavelength_map = custom_get_wavelength_map.call(proc)

Here we have generated a wavelength_map which represent the mapping from
detector pixels (row_idx, col_idx) to wavelength (lambda).


Outlier cleaning
----------------

If outliers remain in your rate-images, then this step offers a method for
cleaning them without using any time-domain information. Outlier cleaning
is performed by finding deviations from an estimated spatial profile of
the spectral trace, following
`Horne 1986 <https://iopscience.iop.org/article/10.1086/131801/meta>`_.
Outliers can also be optionally found by specifying specific data quality
flags you wish to expunge. See
:class:`CleanOutliersStep <exotic_miri.stage_2.CleanOutliersStep>`.

.. code-block:: python

    from exotic_miri.stage_2 import CleanOutliersStep


    custom_clean_outliers = CleanOutliersStep()
    proc, P, O = custom_clean_outliers.call(proc, dq_bits_to_mask=[0, 11],
                                            window_widths=[40],
                                            poly_order=4, outlier_threshold=5.0)

Here, a spatial profile is constructed from fourth-order polynomials, using
windows 40 pixels long in the dispersion direction. Outlying pixels are replaced
if their values are >5 sigma from the spatial profile or if they have data
quality flags 2**0 (do_not_use) or 2**11 (hot pixel). See the data quality flags
`table <https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#data-quality-flags>`_
in the docs.

Also returned is a 3D array of the fitted spatial profiles, and
a count of the number of outliers cleaned within 0-4 pixels of the spectral
trace (column index 36). You can try setting draw_cleaning_col=True to make
some interactive plots and get a better feel for the cleaning process, and help
tailor the parameters to your dataset.


Background subtraction
----------------------

To perform background subtraction from your rate-images, you can make use of
:class:`BackgroundSubtractStep <exotic_miri.stage_2.BackgroundSubtractStep>`.

.. code-block:: python

    from exotic_miri.stage_2 import BackgroundSubtractStep


    custom_bkg_subtract = BackgroundSubtractStep()
    proc = custom_bkg_subtract.call(proc, method="row_wise")

Here, we have used the default background regions either side of the spectral
trace and applied a row-wise background subtraction. There more options for
estimating the background as a linear function of detector column, or for
smoothing over the background. The constant method is not recommended for MIRI
LRS data.


Extract 1D spectra
------------------

To extract a time-series of 1D stellar spectra, using a box aperture, you
can make use of
:class:`Extract1DBoxStep <exotic_miri.stage_2.Extract1DBoxStep>`.

.. code-block:: python

    from exotic_miri.stage_2 import Extract1DBoxStep


    custom_extract1d_box = Extract1DBoxStep()
    wv, spec, spec_unc, trace_sigmas = custom_extract1d_box.call(
        proc, wavelength_map, trace_position=36,
        aperture_center=36, aperture_left_width=4, aperture_right_width=4)

Here, a box aperture (top-hat function) is centred on column 36 (nominal for
MIRI LRS) and extends 4 pixels in each direction. The total aperture is therefore
9 pixels wide. Note that you must have run the GetWavelengthMap step, so that you
may pass the wavelength map as an input. Note that this step returns four outputs:
the wavelengths, the time-series spectra, the uncertainties, and a measure of the
PSF widths.

ExoTiC-MIRI also has an implementation of optimal extraction
`(Horne 1986) <https://iopscience.iop.org/article/10.1086/131801/meta>`_.
See :class:`Extract1DOptimalStep <exotic_miri.stage_2.Extract1DOptimalStep>` for
details.

Align spectra
-------------

Often the pointing stability corresponds to the flux stability in your light curves.
The x and y position of the spectral trace through time may be used as a diagnostic or
decorrelator. To measure the positions, and optionally re-align the spectra,
you can use :class:`AlignSpectraStep <exotic_miri.stage_2.AlignSpectraStep>`.

.. code-block:: python

    from exotic_miri.stage_2 import AlignSpectraStep


    custom_align_spectra = AlignSpectraStep()
    spec, spec_unc, x_shifts, y_shifts = custom_align_spectra.call(
        proc, spec, spec_unc, align_spectra=False)

Note that this step requires the outputs from Extract1DBoxStep or Extract1DOptimalStep as
inputs.

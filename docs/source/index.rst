ExoTiC-MIRI
===========

Custom steps for JWST MIRI LRS data reduction from raw data to light curves.
Interoperable with the STScI pipeline.

The Space Telescope `pipeline <https://github.com/spacetelescope/jwst>`_
for JWST defines two main stages for the processing of raw observational data to
a time series of 1D spectra.
    - Stage 1: starts from the _uncal.fits files and performs basic detector-level
      corrections at the group level. This is followed by ramp fitting to make
      _rateints.fits files, or rate-images. The dimensions of the data are transformed
      from [ints, groups, rows, cols] to [ints, rows, cols].
    - Stage 2: picks up the _rateints.fits files and performs mode-specific
      corrections on these rate-images. This is followed by extraction of the
      spectra to make light curves. The dimensions of the data are transformed
      from [ints, rows, cols] to [ints, wavelength].
In this package, we make available custom steps
that can be swapped in and out of both stage 1 and stage 2 data processing.
The custom steps provided are built specifically for reducing time-series
observations from the Mid-Infrared Instrument's low resolution spectrometer
(MIRI LRS). This mode is of particular use to transiting exoplanet observations,
and as such the algorithms are designed with precise relative fluxes in mind.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   Installation <views/installation>
   Quick start <views/quick_start>
   Stage 1 steps <views/stage_1_steps>
   Stage 2 steps <views/stage_2_steps>
   Complete pipelines <views/complete_pipelines>
   API <views/api/api>
   Citation <views/citation>

Acknowledgements
----------------

Built by David Grant, Daniel Valentine, Hannah Wakeford, and
`contributors <https://github.com/Exo-TiC/ExoTiC-MIRI/graphs/contributors>`_.

If you make use of ExoTiC-MIRI in your research, see the :doc:`citation
page <views/citation>` for info on how to cite this package. You can find
other software from the Exoplanet Timeseries Characterisation (ExoTiC)
ecosystem over on `GitHub <https://github.com/Exo-TiC>`_.

ExoTiC-MIRI
===========

Custom steps for JWST MIRI data reduction. Interoperable with the STScI
pipeline.

The Space Telescope `pipeline <https://github.com/spacetelescope/jwst>`_ for
the James Webb Space Telescope defines two main stages for the processing of
raw observational data to 1d spectra. Stage 1 involves basic detector-level
corrections for individual groups, followed by ramp fitting to make rate
images. Stage 2 involves mode-specific corrections for individual rate images,
followed by 1d extraction to make spectra. In this package we make available
custom steps for both stage 1 and stage 2 data processing.

The custom steps provided are built specifically for reducing time-series
observations from the Mid-Infrared Instrument using when in low resolution
spectroscopic mode. This mode is of particular use to exoplanet observations,
and as such the algorithms are designed with precise relative fluxes in mind.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   Installation <views/installation>
   Quick start <views/quick_start>
   Custom steps <views/custom_steps>
   Citation <views/citation>

Attribution
-----------

TBD.

You can find other software from the Exoplanet Timeseries Characterisation
(ExoTiC) ecosystem over on `GitHub <https://github.com/Exo-TiC>`_.

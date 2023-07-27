ExoTiC-MIRI
===========

Custom steps for JWST MIRI LRS data reduction. Interoperable with the STScI
pipeline. The Space Telescope `pipeline <https://github.com/spacetelescope/jwst>`_
for JWST defines two main stages for the processing of raw observational data to
a time series of 1D spectra.
    - Stage 1: basic detector-level corrections at the group level, followed
      by ramp fitting to make rate-images (_uncal.fits [ints, groups, rows, cols]
      --> _rateints.fits [ints, rows, cols]).
    - Stage 2: mode-specific corrections for the rate-images, followed by
      extraction of the spectra (_rateints.fits [ints, rows, cols]
      -- > time-series spectra [ints, wavelength]).
In this package, we make available custom steps
that can be swapped in and out of both stage 1 and stage 2 data processing.
The custom steps provided are built specifically for reducing time-series
observations from the Mid-Infrared Instrument's low resolution spectrometer
(MIRI LRS). This mode is of particular use to transiting exoplanet observations,
and as such the algorithms are designed with precise relative fluxes in mind.

.. toctree::
   :maxdepth: 2
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

Built by David Grant and
`contributors <https://github.com/Exo-TiC/ExoTiC-MIRI/graphs/contributors>`_.

If you make use of ExoTiC-MIRI in your research, see the :doc:`citation
page <views/citation>` for info on how to cite this package. You can find
other software from the Exoplanet Timeseries Characterisation (ExoTiC)
ecosystem over on `GitHub <https://github.com/Exo-TiC>`_.

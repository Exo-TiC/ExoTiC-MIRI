Stage 1 steps
=============

The stage 1 steps perform basic detector-level corrections at the group
level, followed by ramp fitting. Below we detail the workings of
several steps that you may find useful for improving the quality of your
reduction beyond the default
`JWST stage 1 <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_detector1.html>`_.
For a complete list of ExoTiC-MIRI steps and their workings, see the
:doc:`API <api/api>`.

The following examples assume you have setup the pipelines and loaded
in an _uncal.fits data segment. For example:

.. code-block:: python

    import os

    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    os.environ["CRDS_PATH"] = "path/to/your/local/crds_dir"
    os.environ["CRDS_CONTEXT"] = "jwst_1100.pmap"

    from jwst import datamodels
    from jwst.pipeline import calwebb_detector1

    # Load data segment.
    proc = datamodels.RampModel("path/to/your/data/jwxyz-segnnn_mirimage_uncal.fits")


Drop groups
-----------

The MIRI detector is adversely affected by several effects such as
Reset Switch Charge Decay and dragging down of the final frame.
As such, you may wish to mask certain groups from being included in
subsequent steps, such as jump detection or ramp fitting. This can be
achieved with :class:`DropGroupsStep <exotic_miri.stage_1.DropGroupsStep>`.

.. code-block:: python

    from exotic_miri.stage_1 import DropGroupsStep


    custom_drop_groups = DropGroupsStep()
    proc = custom_drop_groups.call(proc, drop_groups=[0, 1, 2, 3, 4, -1])

Here, we have masked the first 5 groups and the final group. The number of
groups to mask will depend primarily on the brightness of your target, but
in general at least the final group should always be masked (assuming FASTR1
readout mode).


Group-level background subtraction
----------------------------------

If you would like to experiment with subtracting the background/sky at the
group level, to clean up the data before ramp fitting, then you can make use of
:class:`GroupBackgroundSubtractStep <exotic_miri.stage_1.GroupBackgroundSubtractStep>`.

.. code-block:: python

    from exotic_miri.stage_1 import GroupBackgroundSubtractStep


    custom_group_bkg_subtract = GroupBackgroundSubtractStep()
    proc = custom_group_bkg_subtract.call(proc, method="row_wise")

Here, we have used the default background regions either side of the spectral
trace and applied a row-wise background subtraction.

Custom gain
-----------

The default gain value for MIRI may not be the most accurate. This can have
effects on other steps which require the gain to estimate statistical
information. You can create a custom gain model with
:class:`SetCustomGain <exotic_miri.reference.SetCustomGain>`.

.. code-block:: python

    from exotic_miri.reference import SetCustomGain


    custom_set_gain = SetCustomGain()
    stsci_jump = calwebb_detector1.jump_step.JumpStep()
    stsci_ramp_fit = calwebb_detector1.ramp_fit_step.RampFitStep()
    stsci_gain_scale = calwebb_detector1.gain_scale_step.GainScaleStep()

    gain_model = custom_set_gain.call(proc, gain_value=3.1)
    proc = stsci_jump.call(proc, override_gain=gain_model)
    _, proc = stsci_ramp_fit.call(proc, override_gain=gain_model)
    proc = stsci_gain_scale.call(proc, override_gain=gain_model)

Here we have created a custom gain model with a value of 3.1
(see `Bell et al. 2023 <https://arxiv.org/abs/2301.06350>`_) and then
passed this to the jump detection, ramp fitting, and gain scale steps.


Custom linearity correction
---------------------------

Determining a linearity correction, the model which accounts for the
decrease in gain as pixels become increasingly full, is challenging for
MIRI given all the nuances to this Si:As detector. It may be worth generating
a custom linearity correction which is self-calibrated from your dataset,
if you have a sufficient number of groups, using
:class:`SetCustomLinearity <exotic_miri.reference.SetCustomLinearity>`.

.. code-block:: python

    from exotic_miri.reference import SetCustomLinearity


    custom_set_linearity = SetCustomLinearity()
    stsci_linearity = calwebb_detector1.linearity_step.LinearityStep()

    linearity_model = custom_linearity.call(proc, group_idx_start_fit=5, group_idx_end_fit=40,
                                            group_idx_start_derive=5, group_idx_end_derive=100,
                                            row_idx_start_used=350, row_idx_end_used=386)
    proc = stsci_linearity.call(proc, override_linearity=linearity_model)

This correction involves extrapolating a linear fit to an assumed linear, or
well-behaved section of the ramps. In this case, this is between groups 5 and
40. A polynomial is then fit to the ramps for data between groups 5 and 100 and
for rows 350 to 386. The polynomial has the constant- and linear-term coefficients
fixed at 0 and 1, respectively. This polynomial then serves as the correction for
all ramps in all data segments hereafter.

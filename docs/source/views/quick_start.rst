Quick start
===========

After installing the code (see :doc:`installation <installation>`) you
are ready to reduce some data. Below is a minimal demo of how the ExoTiC-MIRI
step may be interleaved with the default JWST pipeline steps.

First, set your CRDS config and then import the JWST pipeline.

.. code-block:: python

    import os

    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    os.environ["CRDS_PATH"] = "path/to/your/local/crds_dir"
    os.environ["CRDS_CONTEXT"] = "jwst_1100.pmap"

    from jwst import datamodels
    from jwst.pipeline import calwebb_detector1
    from jwst.pipeline import calwebb_spec2


    stsci_dq_init = calwebb_detector1.dq_init_step.DQInitStep()
    stsci_saturation = calwebb_detector1.saturation_step.SaturationStep()
    stsci_reset = calwebb_detector1.reset_step.ResetStep()
    stsci_linearity = calwebb_detector1.linearity_step.LinearityStep()
    stsci_dark_current = calwebb_detector1.dark_current_step.DarkCurrentStep()
    stsci_jump = calwebb_detector1.jump_step.JumpStep()
    stsci_ramp_fit = calwebb_detector1.ramp_fit_step.RampFitStep()

Next, import the ExoTiC-MIRI steps you wish to implement. In this example, let's
say you want additional functionality to mask certain groups and subtract the
background at the group level.

.. code-block:: python

    from exotic_miri.stage_1 import DropGroupsStep, GroupBackgroundSubtractStep


    custom_drop_groups = DropGroupsStep()
    custom_group_bkg_subtract = GroupBackgroundSubtractStep()

And so, you may now simply apply a mixture of default and custom steps to your data.
The ExoTiC-MIRI steps work seamlessly with the default steps.

.. code-block:: python

    proc = datamodels.RampModel("path/to/your/data/jwxyz-segnnn_mirimage_uncal.fits")

    proc = stsci_dq_init.call(proc)
    proc = stsci_saturation.call(proc)
    proc = stsci_reset.call(proc)
    proc = custom_drop_groups.call(proc, drop_groups=[0, 1, 2, 3, 4, 5, -1])  # <-- custom step
    proc = stsci_linearity.call(proc)
    proc = stsci_dark_current.call(proc)
    proc = custom_group_bkg_subtract.call(proc, method="row_wise")  # <-- custom step
    proc = stsci_jump.call(proc)
    _, proc = stsci_ramp_fit.call(proc)

That's it. For more information on the custom steps available, please see the
tutorials on :doc:`stage 1 steps <stage_1_steps>`,
:doc:`stage 2 steps <stage_2_steps>`, or the :doc:`API <api/api>`. There are
also :doc:`complete pipelines <complete_pipelines>` which have been employed
on real JWST datasets, which you may find helpful.

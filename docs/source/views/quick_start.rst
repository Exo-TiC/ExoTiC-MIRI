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


    stsci_dark_current = calwebb_detector1.dark_current_step.DarkCurrentStep()
    stsci_jump = calwebb_detector1.jump_step.JumpStep()
    stsci_ramp_fit = calwebb_detector1.ramp_fit_step.RampFitStep()

Next, import the ExoTiC-MIRI steps you wish to implement. In this example, let's
say you want additional functionality to mask certain groups before jump detection
and ramp fitting.

.. code-block:: python

    from exotic_miri.stage_1 import DropGroupsStep


    custom_drop_groups = DropGroupsStep()

You may now simply apply a mixture of default and custom steps to your data.
The ExoTiC-MIRI steps work seamlessly with the default steps.

.. code-block:: python

    proc = datamodels.RampModel("path/to/your/data/jwxyz-segnnn_mirimage_uncal.fits")

    proc = stsci_dark_current.call(proc)
    proc = custom_drop_groups.call(proc, drop_groups=[0, 1, 2, 3, 4, -1])  # <-- custom step
    proc = stsci_jump.call(proc)
    _, proc = stsci_ramp_fit.call(proc)

That's it. For more information on the custom steps available, please see the
:doc:`API <api/api>`. There are also :doc:`complete pipelines <complete_pipelines>`
which have been employed on real JWST datasets which you may find helpful.

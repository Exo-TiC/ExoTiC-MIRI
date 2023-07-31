Complete pipelines
==================

Here we provide complete pipelines that have been applied on real JWST
datasets, to help fast-track your ExoTiC-MIRI implementation. Some of the
steps in the pipeline depend on the target, mainly the brightness, and
therefore the number of groups. Below we show a couple of pipeline options
for datasets with more or less than 40 groups, although please note that
this number is somewhat experimental at this time.

To get these up and running, you will have to take a look through the code
and update the paths specific to your machine, as well as any of the args
specific to the number of groups etc. The processed data is written out at
the end of stage 1, as rate-image fits files, and at the end of stage 2, as
an xarray.

For observations with <40 groups
--------------------------------

Here is an example pipeline for bright targets, e.g., 7 groups.

.. code-block:: python

    import os
    import numpy as np
    import xarray as xr

    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    os.environ["CRDS_PATH"] = "path/to/your/local/crds_dir"
    os.environ["CRDS_CONTEXT"] = "jwst_1100.pmap"

    from jwst import datamodels
    from jwst.pipeline import calwebb_detector1, calwebb_spec2

    from exotic_miri.reference import SetCustomGain, GetWavelengthMap
    from exotic_miri.stage_1 import DropGroupsStep
    from exotic_miri.stage_2 import CleanOutliersStep, BackgroundSubtractStep, \
        Extract1DBoxStep, AlignSpectraStep


    # Data and reduction config.
    seg_names = ["jwxyz-seg001_mirimage", "jwxyz-seg002_mirimage", "jwxyz-seg003_mirimage",
                 "jwxyz-seg004_mirimage", "jwxyz-seg005_mirimage", "jwxyz-seg006_mirimage"]
    data_dir = "/path/to/your/uncal_data_dir"
    reduction_dir = "/path/to/your/reduction_dir"
    stage_1_dir = os.path.join(reduction_dir, "stage_1")
    stage_2_dir = os.path.join(reduction_dir, "stage_2")
    for _dir in [reduction_dir, stage_1_dir, stage_2_dir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    # Instantiate STScI steps for stage 1.
    stsci_group_scale = calwebb_detector1.group_scale_step.GroupScaleStep()
    stsci_dq_init = calwebb_detector1.dq_init_step.DQInitStep()
    stsci_saturation = calwebb_detector1.saturation_step.SaturationStep()
    stsci_reset = calwebb_detector1.reset_step.ResetStep()
    stsci_linearity = calwebb_detector1.linearity_step.LinearityStep()
    stsci_dark_current = calwebb_detector1.dark_current_step.DarkCurrentStep()
    stsci_jump = calwebb_detector1.jump_step.JumpStep()
    stsci_ramp_fit = calwebb_detector1.ramp_fit_step.RampFitStep()
    stsci_gain_scale = calwebb_detector1.gain_scale_step.GainScaleStep()

    # Instantiate custom steps for stage 1.
    custom_set_gain = SetCustomGain()
    custom_drop_groups = DropGroupsStep()

    # Instantiate STScI steps for stage 2.
    stsci_assign_wcs = calwebb_spec2.assign_wcs_step.AssignWcsStep()
    stsci_srctype = calwebb_spec2.srctype_step.SourceTypeStep()
    stsci_flat_field = calwebb_spec2.flat_field_step.FlatFieldStep()

    # Instantiate custom steps for stage 2.
    custom_get_wavelength_map = GetWavelengthMap()
    custom_clean_outliers = CleanOutliersStep()
    custom_background_subtract = BackgroundSubtractStep()
    custom_extract1d_box = Extract1DBoxStep()
    custom_align_spectra = AlignSpectraStep()

    # Make custom gain datamodel.
    uncal_last = datamodels.RampModel(os.path.join(data_dir, "{}_uncal.fits".format(seg_names[-1])))
    gain_model = custom_set_gain.call(uncal_last, gain_value=3.1)
    del uncal_last

    # Iterate data chunks.
    all_spec = []
    all_spec_unc = []
    all_bkg = []
    all_x_shifts = []
    all_y_shifts = []
    all_trace_sigmas = []
    all_O = []
    for seg in seg_names:
        print("\n========= Working on {} =========\n".format(seg))

        # Read in segment.
        proc = datamodels.RampModel(os.path.join(data_dir, "{}_uncal.fits".format(seg)))

        # Stage 1 reduction.
        proc = stsci_group_scale.call(proc)
        proc = stsci_dq_init.call(proc)
        proc = stsci_saturation.call(proc, n_pix_grow_sat=1)
        proc = stsci_reset.call(proc)
        proc = custom_drop_groups.call(proc, drop_groups=[6])
        proc = stsci_linearity.call(proc)
        proc = stsci_dark_current.call(proc)
        proc = stsci_jump.call(
            proc,
            rejection_threshold=15.,
            four_group_rejection_threshold=15.,
            three_group_rejection_threshold=15.,
            flag_4_neighbors=False,
            min_jump_to_flag_neighbors=15.,
            expand_large_events=False,
            skip=False, override_gain=gain_model)
        _, proc = stsci_ramp_fit.call(proc, override_gain=gain_model)
        proc = stsci_gain_scale.call(proc, override_gain=gain_model)
        proc.save(path=os.path.join(stage_1_dir, "{}_stage_1.fits".format(seg)))

        # Stage 2 reduction, part 1: auxiliary data.
        proc = stsci_assign_wcs.call(proc)
        proc = stsci_srctype.call(proc)
        wavelength_map = custom_get_wavelength_map.call(proc)

        # Stage 2 reduction, part 2: cleaning.
        proc = stsci_flat_field.call(proc)
        temp, _, _ = custom_clean_outliers.call(
            proc, dq_bits_to_mask=[0, ],
            window_widths=[150, 100, 50, 40, 30, 24], poly_order=4, outlier_threshold=5.0)
        _, bkg = custom_background_subtract.call(
            temp, method="row_wise", smoothing_length=None,
            bkg_col_left_start=12, bkg_col_left_end=22,
            bkg_col_right_start=50, bkg_col_right_end=68)
        proc.data -= bkg
        proc, P, O = custom_clean_outliers.call(
            proc, dq_bits_to_mask=[0, ],
            window_widths=[150, 100, 50, 40, 30, 24], poly_order=4, outlier_threshold=5.0)

        # Stage 2 reduction, part 3: extraction.
        proc.err[~np.isfinite(proc.err)] = 0.
        wv, spec, spec_unc, trace_sigmas = custom_extract1d_box.call(
            proc, wavelength_map,
            trace_position="constant", aperture_center=36,
            aperture_left_width=4, aperture_right_width=4)
        spec, spec_unc, x_shifts, y_shifts = custom_align_spectra.call(
            proc, spec, spec_unc, align_spectra=False)

        all_spec.append(spec)
        all_spec_unc.append(spec_unc)
        all_bkg.append(np.median(bkg, axis=2))
        all_x_shifts.append(x_shifts)
        all_y_shifts.append(y_shifts)
        all_trace_sigmas.append(trace_sigmas)
        all_O.append(O)

    # Build xarray for all data products.
    ds = xr.Dataset(
        data_vars=dict(
            flux=(["integration_number", "wavelength"], np.concatenate(all_spec), {"units": "DN/s"}),
            flux_error=(["integration_number", "wavelength"], np.concatenate(all_spec_unc), {"units": "DN/s"}),
            background=(["integration_number", "wavelength"], np.concatenate(all_bkg), {"units": "DN/s"}),
            x_shift=(["integration_number"], np.concatenate(all_x_shifts), {"units": "pixel"}),
            y_shift=(["integration_number"], np.concatenate(all_y_shifts), {"units": "pixel"}),
            psf_sigma=(["integration_number"], np.concatenate(all_trace_sigmas), {"units": "pixel"}),
            n_outliers=(["integration_number", "wavelength", "region_width"], np.concatenate(all_O), {"units": ""}),
        ),
        coords=dict(
            integration_number=(["integration_number"], np.arange(1, 1 + np.concatenate(all_spec).shape[0], 1), {"units": ""}),
            wavelength=(["wavelength"], wv, {"units": "microns"}),
            region_width=(["region_width"], np.arange(0, 5, 1), {"units": "pixel"}),
        ),
        attrs=dict(author="Your name",
                   contact="Your email",
                   code="ExoTiC-MIRI interoperating with the STScI pipeline"
                   )
    )
    res_path = os.path.join(
        stage_2_dir, "stage_2_output.nc")
    ds.to_netcdf(res_path)


For observations with >40 groups
--------------------------------

Here is an example pipeline for a dimmer target, e.g., 100 groups. In this pipeline
we implement the self-calibrated linearity correction.

.. code-block:: python

    import os
    import numpy as np
    import xarray as xr

    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    os.environ["CRDS_PATH"] = "path/to/your/local/crds_dir"
    os.environ["CRDS_CONTEXT"] = "jwst_1100.pmap"

    from jwst import datamodels
    from jwst.pipeline import calwebb_detector1, calwebb_spec2

    from exotic_miri.reference import SetCustomGain, SetCustomLinearity, GetWavelengthMap
    from exotic_miri.stage_1 import DropGroupsStep
    from exotic_miri.stage_2 import CleanOutliersStep, BackgroundSubtractStep, \
        Extract1DBoxStep, AlignSpectraStep


    # Data and reduction config.
    seg_names = ["jwxyz-seg001_mirimage", "jwxyz-seg002_mirimage"]
    data_dir = "/path/to/your/uncal_data_dir"
    reduction_dir = "/path/to/your/reduction_dir"
    stage_1_dir = os.path.join(reduction_dir, "stage_1")
    stage_2_dir = os.path.join(reduction_dir, "stage_2")
    for _dir in [reduction_dir, stage_1_dir, stage_2_dir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    # Instantiate STScI steps for stage 1.
    stsci_group_scale = calwebb_detector1.group_scale_step.GroupScaleStep()
    stsci_dq_init = calwebb_detector1.dq_init_step.DQInitStep()
    stsci_saturation = calwebb_detector1.saturation_step.SaturationStep()
    stsci_reset = calwebb_detector1.reset_step.ResetStep()
    stsci_linearity = calwebb_detector1.linearity_step.LinearityStep()
    stsci_dark_current = calwebb_detector1.dark_current_step.DarkCurrentStep()
    stsci_jump = calwebb_detector1.jump_step.JumpStep()
    stsci_ramp_fit = calwebb_detector1.ramp_fit_step.RampFitStep()
    stsci_gain_scale = calwebb_detector1.gain_scale_step.GainScaleStep()

    # Instantiate custom steps for stage 1.
    custom_set_gain = SetCustomGain()
    custom_set_linearity = SetCustomLinearity()
    custom_drop_groups = DropGroupsStep()

    # Instantiate STScI steps for stage 2.
    stsci_assign_wcs = calwebb_spec2.assign_wcs_step.AssignWcsStep()
    stsci_srctype = calwebb_spec2.srctype_step.SourceTypeStep()
    stsci_flat_field = calwebb_spec2.flat_field_step.FlatFieldStep()

    # Instantiate custom steps for stage 2.
    custom_get_wavelength_map = GetWavelengthMap()
    custom_clean_outliers = CleanOutliersStep()
    custom_background_subtract = BackgroundSubtractStep()
    custom_extract1d_box = Extract1DBoxStep()
    custom_align_spectra = AlignSpectraStep()

    # Make custom gain datamodel.
    uncal_last = datamodels.RampModel(os.path.join(data_dir, "{}_uncal.fits".format(seg_names[-1])))
    gain_model = custom_set_gain.call(uncal_last, gain_value=3.1)

    # Make custom linearity model.
    linearity_model = custom_set_linearity.call(
                uncal_last, group_idx_start_fit=10, group_idx_end_fit=40,
                group_idx_start_derive=10, group_idx_end_derive=99,
                row_idx_start_used=300, row_idx_end_used=380)
    del uncal_last

    # Iterate data chunks.
    all_spec = []
    all_spec_unc = []
    all_bkg = []
    all_x_shifts = []
    all_y_shifts = []
    all_trace_sigmas = []
    all_O = []
    for seg in seg_names:
        print("\n========= Working on {} =========\n".format(seg))

        # Read in segment.
        proc = datamodels.RampModel(os.path.join(data_dir, "{}_uncal.fits".format(seg)))

        # Stage 1 reduction.
        proc = stsci_group_scale.call(proc)
        proc = stsci_dq_init.call(proc)
        proc = stsci_saturation.call(proc, n_pix_grow_sat=1)
        proc = stsci_reset.call(proc)
        proc = custom_drop_groups.call(proc, drop_groups=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99])
        proc = stsci_linearity.call(proc, override_linearity=linearity_model)
        proc = stsci_dark_current.call(proc)
        proc = stsci_jump.call(
            proc,
            rejection_threshold=15.,
            four_group_rejection_threshold=15.,
            three_group_rejection_threshold=15.,
            flag_4_neighbors=False,
            min_jump_to_flag_neighbors=15.,
            expand_large_events=False,
            skip=False, override_gain=gain_model)
        _, proc = stsci_ramp_fit.call(proc, override_gain=gain_model)
        proc = stsci_gain_scale.call(proc, override_gain=gain_model)
        proc.save(path=os.path.join(stage_1_dir, "{}_stage_1.fits".format(seg)))

        # Stage 2 reduction, part 1: auxiliary data.
        proc = stsci_assign_wcs.call(proc)
        proc = stsci_srctype.call(proc)
        wavelength_map = custom_get_wavelength_map.call(proc)

        # Stage 2 reduction, part 2: cleaning.
        proc = stsci_flat_field.call(proc)
        temp, _, _ = custom_clean_outliers.call(
            proc, dq_bits_to_mask=[0, ],
            window_widths=[150, 100, 50, 40, 30, 24], poly_order=4, outlier_threshold=5.0)
        _, bkg = custom_background_subtract.call(
            temp, method="row_wise", smoothing_length=None,
            bkg_col_left_start=12, bkg_col_left_end=22,
            bkg_col_right_start=50, bkg_col_right_end=68)
        proc.data -= bkg
        proc, P, O = custom_clean_outliers.call(
            proc, dq_bits_to_mask=[0, ],
            window_widths=[150, 100, 50, 40, 30, 24], poly_order=4, outlier_threshold=5.0)

        # Stage 2 reduction, part 3: extraction.
        proc.err[~np.isfinite(proc.err)] = 0.
        wv, spec, spec_unc, trace_sigmas = custom_extract1d_box.call(
            proc, wavelength_map,
            trace_position="constant", aperture_center=36,
            aperture_left_width=3, aperture_right_width=3)
        spec, spec_unc, x_shifts, y_shifts = custom_align_spectra.call(
            proc, spec, spec_unc, align_spectra=False)

        all_spec.append(spec)
        all_spec_unc.append(spec_unc)
        all_bkg.append(np.median(bkg, axis=2))
        all_x_shifts.append(x_shifts)
        all_y_shifts.append(y_shifts)
        all_trace_sigmas.append(trace_sigmas)
        all_O.append(O)

    # Build xarray for all data products.
    ds = xr.Dataset(
        data_vars=dict(
            flux=(["integration_number", "wavelength"], np.concatenate(all_spec), {"units": "DN/s"}),
            flux_error=(["integration_number", "wavelength"], np.concatenate(all_spec_unc), {"units": "DN/s"}),
            background=(["integration_number", "wavelength"], np.concatenate(all_bkg), {"units": "DN/s"}),
            x_shift=(["integration_number"], np.concatenate(all_x_shifts), {"units": "pixel"}),
            y_shift=(["integration_number"], np.concatenate(all_y_shifts), {"units": "pixel"}),
            psf_sigma=(["integration_number"], np.concatenate(all_trace_sigmas), {"units": "pixel"}),
            n_outliers=(["integration_number", "wavelength", "region_width"], np.concatenate(all_O), {"units": ""}),
        ),
        coords=dict(
            integration_number=(["integration_number"], np.arange(1, 1 + np.concatenate(all_spec).shape[0], 1), {"units": ""}),
            wavelength=(["wavelength"], wv, {"units": "microns"}),
            region_width=(["region_width"], np.arange(0, 5, 1), {"units": "pixel"}),
        ),
        attrs=dict(author="Your name",
                   contact="Your email",
                   code="ExoTiC-MIRI interoperating with the STScI pipeline"
                   )
    )
    res_path = os.path.join(
        stage_2_dir, "stage_2_output.nc")
    ds.to_netcdf(res_path)


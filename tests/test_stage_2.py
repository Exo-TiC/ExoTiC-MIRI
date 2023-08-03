import os
import json
import unittest
import numpy as np

with open("config.json", "r") as read_file:
    config = json.load(read_file)

os.environ["CRDS_SERVER_URL"] = config["CRDS_SERVER"]
os.environ["CRDS_PATH"] = config["CRDS_DIR"]
os.environ["CRDS_CONTEXT"] = config["CRDS_PMAP"]

from jwst import datamodels
from jwst.pipeline import calwebb_spec2
from exotic_miri.reference import GetWavelengthMap
from exotic_miri.stage_2 import InspectDQFlagsStep, CleanOutliersStep, \
    BackgroundSubtractStep, Extract1DBoxStep, Extract1DOptimalStep, AlignSpectraStep


class TestStageOne(unittest.TestCase):
    """ Test exotic_miri.stage_1 subpackage. """

    def __init__(self, *args, **kwargs):
        super(TestStageOne, self).__init__(*args, **kwargs)
        self.cube_model = None

    def generate_mock_miri_lrs_rateimage_data(self, n_integrations=10,
                                              n_rows=416, n_cols=72):
        # Build template RampModel data structure.
        self.cube_model = datamodels.CubeModel()
        self.cube_model.data = np.ones((n_integrations, n_rows, n_cols)) * 1.1
        self.cube_model.err = np.ones((n_integrations, n_rows, n_cols)) * 0.1
        self.cube_model.dq = np.zeros((n_integrations, n_rows, n_cols))
        self.cube_model.var_poisson = np.ones((n_integrations, n_rows, n_cols)) * 0.09
        self.cube_model.var_rnoise = np.ones((n_integrations, n_rows, n_cols)) * 0.01

        # Add timing and meta data.
        self._generate_mock_timing_and_meta_data(self.cube_model, n_integrations, 50)

    def _generate_mock_timing_and_meta_data(self, model, n_integrations, n_groups):
        # Add timing data.
        int_times = np.rec.array(
            [(t_idx, t, t, t, t, t, t) for t_idx, t
             in enumerate(np.linspace(2460018.30, 2460018.31, n_integrations))],
            formats="float64,float64,float64,float64,float64,float64,float64",
            names="integration_number,"
                  "int_start_mjd_utc,int_mid_mjd_utc,int_end_mjd_utc,"
                  "int_start_bjd_tdb,int_mid_bjd_tdb,int_end_bjd_tdb")
        model.int_times = int_times

        # Add meta data.
        model.meta.ngroups = n_groups
        model.meta.ngroups_file = n_groups
        model.meta.nints = n_integrations
        model.meta.nints_file = n_integrations
        model.meta.observation.date = "2023-03-13"
        model.meta.observation.time = "11:17:21.976"
        model.meta.exposure.integration_time = 0.16 * n_groups
        model.meta.exposure.ngroups = n_groups
        model.meta.exposure.nints = n_integrations
        model.meta.exposure.type = "MIR_LRS-SLITLESS"
        model.meta.exposure.readpatt = "FASTR1"
        model.meta.instrument.name = "MIRI"
        model.meta.instrument.detector = "MIRIMAGE"
        model.meta.subarray.xstart = 1
        model.meta.subarray.ystart = 529
        model.meta.subarray.xsize = 72
        model.meta.subarray.ysize = 416
        model.meta.wcsinfo.v2_ref = -378.630313
        model.meta.wcsinfo.v3_ref = -344.895045
        model.meta.wcsinfo.roll_ref = 283.9573081811109
        model.meta.wcsinfo.dec_ref = -28.06179623774727
        model.meta.wcsinfo.ra_ref = 239.9622955846874

    def test_inspect_dq_flags(self):
        """ Test inspect DQ flags. """
        self.generate_mock_miri_lrs_rateimage_data()
        custom_inspect_dq_flags = InspectDQFlagsStep()
        custom_inspect_dq_flags.call(self.cube_model, draw_dq_flags=False)

    def test_clean_outliers_step(self):
        """ Test clean outliers step. """
        self.generate_mock_miri_lrs_rateimage_data(n_integrations=3)
        outlier_int_idx = 1
        outlier_row_idx = 200
        outlier_col_idx = 35
        self.cube_model.data[outlier_int_idx, outlier_row_idx, outlier_col_idx] = 5.

        custom_clean_outliers = CleanOutliersStep()
        for dq_m in [[], [0,], [0, 2]]:
            for ww in [[100], [150, 100, 50, 50, 20, 20, 20]]:
                for po in [1, 3]:
                    self.cube_model, P, O = custom_clean_outliers.call(
                        self.cube_model, dq_bits_to_mask=dq_m,
                        window_heights=ww, poly_order=po, outlier_threshold=5.0)

                    self.assertIsInstance(self.cube_model, datamodels.CubeModel)
                    self.assertIsInstance(P, np.ndarray)
                    self.assertIsInstance(O, np.ndarray)

                    self.assertEqual(self.cube_model.shape, (3, 416, 72))
                    self.assertEqual(P.shape, (3, 416, 72))
                    self.assertEqual(O.shape, (3, 416, 5))

                    self.assertEqual(O[outlier_int_idx, outlier_row_idx, 0], 0)
                    self.assertEqual(O[outlier_int_idx, outlier_row_idx, 1], 1)
                    self.assertEqual(O[outlier_int_idx, outlier_row_idx, 2], 1)
                    self.assertEqual(O[outlier_int_idx, outlier_row_idx, 3], 1)
                    self.assertEqual(O[outlier_int_idx, outlier_row_idx, 4], 1)

    def test_bkg_subtract_step(self):
        """ Test background subtract step. """
        self.generate_mock_miri_lrs_rateimage_data()
        bkg_signal = 3.5
        trace_signal = 10.
        self.cube_model.data[:, :, :] = 0.
        self.cube_model.data[:, :, 33:40] += trace_signal
        no_bkg_data = np.copy(self.cube_model.data)
        self.cube_model.data[:, :, :] += bkg_signal

        custom_bkg_subtract = BackgroundSubtractStep()
        for m in ["constant", "row_wise", "col_wise"]:
            for sl in [None, 20]:
                for bkg_region_idxs in [(8, 17, 56, 72), (12, 16, 52, 68)]:
                    _cube_model, bkg = custom_bkg_subtract.call(
                        self.cube_model, method=m,
                        bkg_col_left_start=bkg_region_idxs[0],
                        bkg_col_left_end=bkg_region_idxs[1],
                        bkg_col_right_start=bkg_region_idxs[2],
                        bkg_col_right_end=bkg_region_idxs[3], smoothing_length=sl)

                    self.assertIsInstance(bkg, np.ndarray)
                    self.assertIsInstance(_cube_model, datamodels.CubeModel)

                    np.testing.assert_almost_equal(bkg, 3.5, decimal=6)
                    np.testing.assert_almost_equal(_cube_model.data - no_bkg_data,
                                                   0., decimal=6)

    def test_extract1d_box_step(self):
        """ Test extract 1D box step. """
        self.generate_mock_miri_lrs_rateimage_data()
        bkg_signal = 3.5
        trace_signal = 10.
        self.cube_model.data[:, :, :] = 0.
        self.cube_model.data[:, :, 33:40] += trace_signal
        self.cube_model.data[:, :, :] += bkg_signal

        stsci_assign_wcs = calwebb_spec2.assign_wcs_step.AssignWcsStep()
        stsci_srctype = calwebb_spec2.srctype_step.SourceTypeStep()
        self.cube_model = stsci_srctype.call(self.cube_model)
        self.cube_model = stsci_assign_wcs.call(self.cube_model)
        custom_get_wavelength_map = GetWavelengthMap()
        wavelength_map = custom_get_wavelength_map.call(
            self.cube_model, trim_col_start=0, trim_col_end=73)

        custom_extract1d_box = Extract1DBoxStep()
        for tp in ["constant", "gaussian_fits"]:
            for ap_idxs in [[36, 4, 4], [35, 5, 5], [36, 7, 6]]:

                wv, spec, spec_unc, trace_sigmas = custom_extract1d_box.call(
                    self.cube_model, wavelength_map,
                    trace_position=tp, aperture_center=ap_idxs[0],
                    aperture_left_width=ap_idxs[1], aperture_right_width=ap_idxs[2])

                self.assertIsInstance(wv, np.ndarray)
                self.assertIsInstance(spec, np.ndarray)
                self.assertIsInstance(spec_unc, np.ndarray)
                self.assertIsInstance(trace_sigmas, np.ndarray)

                self.assertEqual(wv.shape, (416,))
                self.assertEqual(spec.shape, (10, 416,))
                self.assertEqual(spec_unc.shape, (10, 416,))
                self.assertEqual(trace_sigmas.shape, (10,))

    def test_extract1d_optimal_step(self):
        """ Test extract 1D optimal step. """
        self.generate_mock_miri_lrs_rateimage_data()
        bkg_signal = 3.5
        trace_signal = 10.
        self.cube_model.data[:, :, :] = 0.
        self.cube_model.data[:, :, 33:40] += trace_signal
        self.cube_model.data[:, :, :] += bkg_signal

        stsci_assign_wcs = calwebb_spec2.assign_wcs_step.AssignWcsStep()
        stsci_srctype = calwebb_spec2.srctype_step.SourceTypeStep()
        self.cube_model = stsci_srctype.call(self.cube_model)
        self.cube_model = stsci_assign_wcs.call(self.cube_model)
        custom_get_wavelength_map = GetWavelengthMap()
        wavelength_map = custom_get_wavelength_map.call(
            self.cube_model, trim_col_start=0, trim_col_end=73)

        custom_extract1d_optimal = Extract1DOptimalStep()
        for msp in [False, True]:
            for ap_idxs in [[36, 6, 6], [35, 7, 7], [36, 12, 13]]:

                wv, spec, spec_unc, trace_sigmas = custom_extract1d_optimal.call(
                    self.cube_model, wavelength_map,
                    self.cube_model.data, np.zeros_like(self.cube_model.data),
                    median_spatial_profile=msp, aperture_center=ap_idxs[0],
                    aperture_left_width=ap_idxs[1], aperture_right_width=ap_idxs[2])

                self.assertIsInstance(wv, np.ndarray)
                self.assertIsInstance(spec, np.ndarray)
                self.assertIsInstance(spec_unc, np.ndarray)
                self.assertIsInstance(trace_sigmas, np.ndarray)

                self.assertEqual(wv.shape, (416,))
                self.assertEqual(spec.shape, (10, 416,))
                self.assertEqual(spec_unc.shape, (10, 416,))
                self.assertEqual(trace_sigmas.shape, (10,))

    def test_align_spectra_step(self):
        """ Test align spectra step. """
        self.generate_mock_miri_lrs_rateimage_data()
        bkg_signal = 3.5
        trace_signal = 10.
        self.cube_model.data[:, :, :] = 0.
        self.cube_model.data[:, :, 33:40] += trace_signal
        self.cube_model.data[:, :, :] += bkg_signal

        stsci_assign_wcs = calwebb_spec2.assign_wcs_step.AssignWcsStep()
        stsci_srctype = calwebb_spec2.srctype_step.SourceTypeStep()
        self.cube_model = stsci_srctype.call(self.cube_model)
        self.cube_model = stsci_assign_wcs.call(self.cube_model)
        custom_get_wavelength_map = GetWavelengthMap()
        wavelength_map = custom_get_wavelength_map.call(
            self.cube_model, trim_col_start=0, trim_col_end=73)

        custom_extract1d_box = Extract1DBoxStep()
        wv, spec, spec_unc, trace_sigmas = custom_extract1d_box.call(
            self.cube_model, wavelength_map,
            trace_position="constant", aperture_center=36,
            aperture_left_width=4, aperture_right_width=4)

        custom_align_spectra = AlignSpectraStep()
        for aligns in [False, True]:
            spec, spec_unc, x_shifts, y_shifts = custom_align_spectra.call(
                self.cube_model, spec, spec_unc, align_spectra=aligns)

            self.assertIsInstance(spec, np.ndarray)
            self.assertIsInstance(spec_unc, np.ndarray)
            self.assertIsInstance(x_shifts, np.ndarray)
            self.assertIsInstance(y_shifts, np.ndarray)

            self.assertEqual(spec.shape, (10, 416,))
            self.assertEqual(spec_unc.shape, (10, 416,))
            self.assertEqual(x_shifts.shape, (10,))
            self.assertEqual(y_shifts.shape, (10,))


if __name__ == "__main__":
    unittest.main()

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
from jwst.pipeline import calwebb_detector1, calwebb_spec2
from exotic_miri.reference import GetIntegrationTimes, SetCustomGain, \
    SetCustomLinearity, GetWavelengthMap, GetDefaultGain, GetDefaultReadNoise


class TestReference(unittest.TestCase):
    """ Test exotic_miri.reference subpackage. """

    def __init__(self, *args, **kwargs):
        super(TestReference, self).__init__(*args, **kwargs)
        self.ramp_model = None
        self.cube_model = None

    def generate_mock_miri_lrs_uncal_data(self, n_integrations=10, n_groups=50,
                                          n_rows=416, n_cols=72):
        # Build template RampModel data structure.
        self.ramp_model = datamodels.RampModel()
        self.ramp_model.data = np.ones((n_integrations, n_groups, n_rows, n_cols)) \
            * np.linspace(1, 10, n_groups)[np.newaxis, :, np.newaxis, np.newaxis]
        self.ramp_model.err = np.ones((n_integrations, n_groups, n_rows, n_cols))
        self.ramp_model.groupdq = np.zeros((n_integrations, n_groups, n_rows, n_cols))
        self.ramp_model.pixeldq = np.zeros((n_rows, n_cols))

        # Add timing and meta data.
        self._generate_mock_timing_and_meta_data(self.ramp_model, n_integrations, n_groups)

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

    def test_get_integration_times(self):
        """ Test get integration times. """
        self.generate_mock_miri_lrs_uncal_data()
        custom_get_integration_times = GetIntegrationTimes()
        timing_data, int_time_s = custom_get_integration_times.call(self.ramp_model)

        self.assertIsInstance(timing_data["int_mid_BJD_TDB"], np.ndarray)
        self.assertIsInstance(int_time_s, float)

    def test_set_custom_gain(self):
        """ Test set custom gain. """
        self.generate_mock_miri_lrs_uncal_data()
        custom_set_gain = SetCustomGain()
        gain_model = custom_set_gain.call(self.ramp_model, gain_value=3.1)

        self.assertIsInstance(gain_model, datamodels.GainModel)
        self.assertEqual(gain_model.data.shape, (1024, 1032))
        self.assertEqual(gain_model.data[0, 0], np.float32(3.1))

    def test_set_custom_linearity(self):
        """ Test set custom linearity. """
        self.generate_mock_miri_lrs_uncal_data()
        custom_set_linearity = SetCustomLinearity()
        linearity_model = custom_set_linearity.call(
            self.ramp_model, group_idx_start_fit=1, group_idx_end_fit=20,
            group_idx_start_derive=1, group_idx_end_derive=49,
            row_idx_start_used=300, row_idx_end_used=380,
            draw_corrections=False)

        self.assertIsInstance(linearity_model, datamodels.LinearityModel)
        self.assertEqual(linearity_model.coeffs.shape, (5, 1024, 1032))
        self.assertEqual(linearity_model.coeffs[0, 50, 50], np.float32(0.))
        self.assertEqual(linearity_model.coeffs[1, 50, 50], np.float32(1.))

    def test_get_wavelength_map(self):
        """ Test get wavelength map. """
        self.generate_mock_miri_lrs_rateimage_data()
        stsci_assign_wcs = calwebb_spec2.assign_wcs_step.AssignWcsStep()
        stsci_srctype = calwebb_spec2.srctype_step.SourceTypeStep()
        self.cube_model = stsci_srctype.call(self.cube_model)
        self.cube_model = stsci_assign_wcs.call(self.cube_model)

        custom_get_wavelength_map = GetWavelengthMap()
        wavelength_map = custom_get_wavelength_map.call(
            self.cube_model, trim_col_start=0, trim_col_end=73)

        self.assertIsInstance(wavelength_map, np.ndarray)
        self.assertEqual(wavelength_map.shape, (416, 72))
        self.assertAlmostEqual(wavelength_map[148, 36], 12.0, places=1)
        self.assertAlmostEqual(wavelength_map[382, 36], 5.0, places=1)

    def test_get_default_gain(self):
        """ Test get default gain. """
        self.generate_mock_miri_lrs_uncal_data()
        default_get_gain = GetDefaultGain()
        gain_model = default_get_gain.call(
            self.ramp_model, median_value=False)

        self.assertIsInstance(gain_model, datamodels.GainModel)
        self.assertEqual(gain_model.data.shape, (1024, 1032))

        gain_value = default_get_gain.call(
            self.ramp_model, median_value=True)

        self.assertIsInstance(gain_value, float)

    def test_get_default_readnoise(self):
        """ Test get default readnoise. """
        self.generate_mock_miri_lrs_uncal_data()
        default_get_readnoise = GetDefaultReadNoise()
        readnoise_model = default_get_readnoise.call(
            self.ramp_model, median_value=False)

        self.assertIsInstance(readnoise_model, datamodels.ReadnoiseModel)
        self.assertEqual(readnoise_model.data.shape, (1024, 1032))

        readnoise_model = default_get_readnoise.call(
            self.ramp_model, median_value=True)

        self.assertIsInstance(readnoise_model, float)


if __name__ == "__main__":
    unittest.main()

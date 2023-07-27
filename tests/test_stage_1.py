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
from jwst.pipeline import calwebb_detector1
from exotic_miri.stage_1 import DropGroupsStep, ReferencePixelStep, \
    GroupBackgroundSubtractStep, DropIntegrationsStep, RegroupStep


class TestStageOne(unittest.TestCase):
    """ Test exotic_miri.stage_1 subpackage. """

    def __init__(self, *args, **kwargs):
        super(TestStageOne, self).__init__(*args, **kwargs)
        self.ramp_model = None

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

    def test_drop_groups_step(self):
        """ Test drop groups step. """
        self.generate_mock_miri_lrs_uncal_data()
        custom_drop_groups = DropGroupsStep()
        self.ramp_model = custom_drop_groups.call(self.ramp_model, drop_groups=[0, 1, 49])

        self.assertIsInstance(self.ramp_model, datamodels.RampModel)
        self.assertIsInstance(self.ramp_model.groupdq, np.ndarray)
        self.assertTrue(np.all(self.ramp_model.groupdq[:, 0, :, :] == 2**0))
        self.assertTrue(np.all(self.ramp_model.groupdq[:, 1, :, :] == 2**0))
        self.assertTrue(np.all(self.ramp_model.groupdq[:, 49, :, :] == 2**0))
        self.assertTrue(np.all(self.ramp_model.groupdq[:, 2:49, :, :] == 0))

    def test_reference_pixel_step(self):
        """ Test reference pixel step. """
        self.generate_mock_miri_lrs_uncal_data()
        amplifier_ref_signal_diffs = [2.1, 2.2, 3.1, 3.2]
        for amplifier_idx, amp_signal in enumerate(amplifier_ref_signal_diffs):
            ref_pix_signal_ampx = np.linspace(0, amp_signal, 50)[np.newaxis, :, np.newaxis]
            self.ramp_model.data[:, :, :, amplifier_idx] = ref_pix_signal_ampx
        before_corr = np.copy(self.ramp_model.data)

        custom_reference_pixel = ReferencePixelStep()
        for sl, oe in zip([None, 20, None, 20], [True, True, False, False]):
            self.ramp_model = custom_reference_pixel.call(
                self.ramp_model, smoothing_length=sl, odd_even_rows=oe)

            self.assertIsInstance(self.ramp_model, datamodels.RampModel)
            for amplifier_idx, amp_signal in enumerate(amplifier_ref_signal_diffs):
                first_group_corr = before_corr[:, 0, :, amplifier_idx::4] \
                                   - self.ramp_model.data[:, 0, :, amplifier_idx::4]
                last_group_corr = before_corr[:, -1, :, amplifier_idx::4] \
                                  - self.ramp_model.data[:, -1, :, amplifier_idx::4]

                np.testing.assert_almost_equal(first_group_corr, 0., decimal=6)
                np.testing.assert_almost_equal(last_group_corr, amp_signal, decimal=6)

    def test_group_bkg_subtract_step(self):
        """ Test group level background subtract step. """
        self.generate_mock_miri_lrs_uncal_data()
        bkg_signal = np.linspace(0, 3.5, 50)[np.newaxis, :, np.newaxis, np.newaxis]
        trace_signal = np.linspace(1, 10, 50)[np.newaxis, :, np.newaxis, np.newaxis]
        self.ramp_model.data[:, :, :, :] = 0.
        self.ramp_model.data[:, :, :, 33:40] += trace_signal
        no_bkg_data = np.copy(self.ramp_model.data)
        self.ramp_model.data[:, :, :, :] += bkg_signal
        
        custom_group_bkg_subtract = GroupBackgroundSubtractStep()
        for m in ["constant", "row_wise", "col_wise"]:
            for sl in [None, 20]:
                for bkg_region_idxs in [(8, 17, 56, 72), (12, 16, 52, 68)]:

                    _ramp_model = custom_group_bkg_subtract.call(
                        self.ramp_model, method=m,
                        bkg_col_left_start=bkg_region_idxs[0],
                        bkg_col_left_end=bkg_region_idxs[1],
                        bkg_col_right_start=bkg_region_idxs[2],
                        bkg_col_right_end=bkg_region_idxs[3], smoothing_length=sl)

                    self.assertIsInstance(_ramp_model, datamodels.RampModel)
                    np.testing.assert_almost_equal(_ramp_model.data - no_bkg_data,
                                                   0., decimal=6)

    def test_drop_integrations_step(self):
        """ Test drop integrations step. """
        self.generate_mock_miri_lrs_uncal_data()
        custom_drop_integrations = DropIntegrationsStep()
        self.ramp_model = custom_drop_integrations.call(self.ramp_model, drop_integrations=[0, 5, 9])

        self.assertIsInstance(self.ramp_model, datamodels.RampModel)
        self.assertEqual(self.ramp_model.data.shape, (7, 50, 416, 72))
        self.assertEqual(self.ramp_model.err.shape, (7, 50, 416, 72))
        self.assertEqual(self.ramp_model.groupdq.shape, (7, 50, 416, 72))
        self.assertEqual(self.ramp_model.pixeldq.shape, (416, 72))

    def test_regroup_step(self):
        """ Test regroup step. """
        self.generate_mock_miri_lrs_uncal_data(n_groups=12)
        custom_regroup = RegroupStep()
        self.ramp_model = custom_regroup.call(self.ramp_model, n_groups=4)

        self.assertIsInstance(self.ramp_model, datamodels.RampModel)
        self.assertEqual(self.ramp_model.data.shape, (30, 4, 416, 72))
        self.assertEqual(self.ramp_model.err.shape, (30, 4, 416, 72))
        self.assertEqual(self.ramp_model.groupdq.shape, (30, 4, 416, 72))
        self.assertEqual(self.ramp_model.pixeldq.shape, (416, 72))


if __name__ == "__main__":
    unittest.main()

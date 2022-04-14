import os
import unittest
import numpy as np
from jwst import datamodels

from exotic_miri import DropGroupsStep


class TestDropGroups(unittest.TestCase):
    """ Test drop groups step. """

    def __init__(self, *args, **kwargs):
        super(TestDropGroups, self).__init__(*args, **kwargs)

        # Build test RampModel data structure.
        self.test_ramp_model = datamodels.RampModel()
        self.test_ramp_model.data = np.ones((100, 50, 416, 72)) * np.linspace(
            1, 10, 50)[np.newaxis, :, np.newaxis, np.newaxis]
        self.test_ramp_model.err = np.ones((100, 50, 416, 72))
        self.test_ramp_model.groupdq = np.zeros((100, 50, 416, 72))
        self.test_ramp_model.pixeldq = np.zeros((416, 72))
        self.test_ramp_model.meta.ngroups = 50
        self.test_ramp_model.meta.ngroups_file = 50
        self.test_ramp_model.meta.nints = 1000
        self.test_ramp_model.meta.nints_file = 1000
        self.test_ramp_model.meta.exposure.integration_time = 10.3376
        self.test_ramp_model.meta.exposure.ngroups = 50
        self.test_ramp_model.meta.exposure.nints = 1000
        self.test_ramp_model.meta.exposure.type = 'MIR_LRS-SLITLESS'

    def test_template(self):
        """ Test template. """
        # Run github actions CI tests.
        self.assertEqual(type(self.test_ramp_model),
                         type(self.test_ramp_model))

    def test_drop_groups(self):
        """ Test the drop group step with valid args. """
        drop_groups = [[0, 1, 2, 3, 49], [0], [49], [10, 20, 30]]
        exp_integration_times = 10.3376 / 50 * (50 - np.array([5, 1, 1, 0]))
        for dg, eit in zip(drop_groups, exp_integration_times):
            dg_ramp_model = DropGroupsStep().call(
                self.test_ramp_model, drop_groups=dg)

            exp_n_groups = 50 - len(dg)
            self.assertEqual(type(self.test_ramp_model),
                             type(dg_ramp_model))
            self.assertEqual(dg_ramp_model.data.shape,
                             (100, exp_n_groups, 416, 72))
            self.assertEqual(dg_ramp_model.err.shape,
                             (100, exp_n_groups, 416, 72))
            self.assertEqual(dg_ramp_model.groupdq.shape,
                             (100, exp_n_groups, 416, 72))
            self.assertEqual(dg_ramp_model.pixeldq.shape,
                             (416, 72))

            self.assertEqual(dg_ramp_model.meta.ngroups, exp_n_groups)
            self.assertEqual(dg_ramp_model.meta.nints, 1000)
            self.assertAlmostEqual(dg_ramp_model.meta.exposure.integration_time,
                                   eit, delta=1e-7)

    def test_drop_groups_invalid_list(self):
        """ Test the drop group step with invalid args. """
        drop_groups = [[-1, 0, 1, 2], [49, 50]]
        for dg in drop_groups:
            dg_ramp_model = DropGroupsStep().call(
                self.test_ramp_model, drop_groups=dg)

            self.assertEqual(type(self.test_ramp_model),
                             type(dg_ramp_model))
            self.assertEqual(dg_ramp_model.meta.cal_step.drop_groups,
                             'SKIPPED')

    def test_regroup_incorrect_data_model(self):
        """ Test the regroup step with incorrect data model. """
        test_model = datamodels.CubeModel()
        test_model.meta.exposure.type = 'MIR_LRS-SLITLESS'
        dg_ramp_model = DropGroupsStep().call(
            test_model, drop_groups=[0, 1, 2, 3, 4, 49])

        self.assertEqual(type(test_model), type(dg_ramp_model))
        self.assertEqual(dg_ramp_model.meta.cal_step.drop_groups,
                         'SKIPPED')

    def test_regroup_unsupported_mode(self):
        """ Test the regroup step with unsupported mode. """
        test_model = datamodels.RampModel()
        test_model.meta.exposure.type = 'NIS_SOSS'
        dg_ramp_model = DropGroupsStep().call(
            test_model, drop_groups=[0, 1, 2, 3, 4, 49])

        self.assertEqual(type(test_model), type(dg_ramp_model))
        self.assertEqual(dg_ramp_model.meta.cal_step.drop_groups,
                         'SKIPPED')


if __name__ == '__main__':
    unittest.main()

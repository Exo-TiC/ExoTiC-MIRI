import unittest
import numpy as np
from jwst import datamodels

from exotic_miri import RegroupStep


class TestRegroup(unittest.TestCase):
    """ Test regroup step. """

    def __init__(self, *args, **kwargs):
        super(TestRegroup, self).__init__(*args, **kwargs)

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

    def test_regroup_multiple(self):
        """ Test the regroup step with a multiple. """
        n_re_groups = [2, 5, 10, 25]
        exp_ints = [2500, 1000, 500, 200]
        for nrg, epi in zip(n_re_groups, exp_ints):
            regrouped_ramp_model = RegroupStep().call(
                self.test_ramp_model, n_groups=nrg)

            self.assertEqual(type(self.test_ramp_model),
                             type(regrouped_ramp_model))
            self.assertEqual(regrouped_ramp_model.data.shape,
                             (epi, nrg, 416, 72))
            self.assertEqual(regrouped_ramp_model.err.shape,
                             (epi, nrg, 416, 72))
            self.assertEqual(regrouped_ramp_model.groupdq.shape,
                             (epi, nrg, 416, 72))
            self.assertEqual(regrouped_ramp_model.pixeldq.shape,
                             (416, 72))

            self.assertEqual(regrouped_ramp_model.meta.ngroups, nrg)
            self.assertEqual(regrouped_ramp_model.meta.nints, epi)
            self.assertEqual(regrouped_ramp_model.meta.exposure
                             .integration_time, 10.3376 * (100 / epi))

    def test_regroup_non_multiple(self):
        """ Test the regroup step with a non-multiple. """
        n_re_groups = [9, 13, 100]
        for nrg in n_re_groups:
            regrouped_ramp_model = RegroupStep().call(
                self.test_ramp_model, n_groups=nrg)

            self.assertEqual(type(self.test_ramp_model),
                             type(regrouped_ramp_model))
            self.assertEqual(regrouped_ramp_model.meta.cal_step.regroup,
                             'SKIPPED')

    def test_regroup_incorrect_data_model(self):
        """ Test the regroup step with incorrect data model. """
        test_model = datamodels.CubeModel()
        test_model.meta.exposure.type = 'MIR_LRS-SLITLESS'
        regrouped_ramp_model = RegroupStep().call(
            test_model, n_groups=10)

        self.assertEqual(type(test_model), type(regrouped_ramp_model))
        self.assertEqual(regrouped_ramp_model.meta.cal_step.regroup,
                         'SKIPPED')

    def test_regroup_unsupported_mode(self):
        """ Test the regroup step with unsupported mode. """
        test_model = datamodels.RampModel()
        test_model.meta.exposure.type = 'NIS_SOSS'
        regrouped_ramp_model = RegroupStep().call(
            test_model, n_groups=10)

        self.assertEqual(type(test_model), type(regrouped_ramp_model))
        self.assertEqual(regrouped_ramp_model.meta.cal_step.regroup,
                         'SKIPPED')


if __name__ == '__main__':
    unittest.main()

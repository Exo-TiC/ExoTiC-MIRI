import os
import unittest
import numpy as np

os.environ['CRDS_PATH'] = 'crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst import datamodels

from exotic_miri import DropIntegrationsStep


class TestDropIntegrations(unittest.TestCase):
    """ Test drop integrations step. """

    def __init__(self, *args, **kwargs):
        super(TestDropIntegrations, self).__init__(*args, **kwargs)

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

    def test_drop_integrations(self):
        """ Test the drop integrations step with valid args. """
        drop_integrations = [[0], [0, 1, 2, 3], [0, 99], [10, 40, 71]]
        for di in drop_integrations:
            di_ramp_model = DropIntegrationsStep().call(
                self.test_ramp_model, drop_integrations=di)

            exp_n_integrations = 100 - len(di)
            self.assertEqual(type(self.test_ramp_model),
                             type(di_ramp_model))
            self.assertEqual(di_ramp_model.data.shape,
                             (exp_n_integrations, 50, 416, 72))
            self.assertEqual(di_ramp_model.err.shape,
                             (exp_n_integrations, 50, 416, 72))
            self.assertEqual(di_ramp_model.groupdq.shape,
                             (exp_n_integrations, 50, 416, 72))
            self.assertEqual(di_ramp_model.pixeldq.shape,
                             (416, 72))

            self.assertEqual(di_ramp_model.meta.ngroups, 50)
            self.assertEqual(di_ramp_model.meta.nints, exp_n_integrations)
            self.assertEqual(di_ramp_model.meta.exposure.integration_time,
                             10.3376)

    def test_drop_groups_invalid_list(self):
        """ Test the drop group step with invalid args. """
        drop_integrations = [[-1, 0, 1, 2], [99, 100]]
        for di in drop_integrations:
            di_ramp_model = DropIntegrationsStep().call(
                self.test_ramp_model, drop_integrations=di)

            self.assertEqual(type(self.test_ramp_model),
                             type(di_ramp_model))
            self.assertEqual(di_ramp_model.meta.cal_step.drop_integrations,
                             'SKIPPED')

    def test_regroup_incorrect_data_model(self):
        """ Test the regroup step with incorrect data model. """
        test_model = datamodels.CubeModel()
        test_model.meta.exposure.type = 'MIR_LRS-SLITLESS'
        di_ramp_model = DropIntegrationsStep().call(
            test_model, drop_integrations=[0])

        self.assertEqual(type(test_model), type(di_ramp_model))
        self.assertEqual(di_ramp_model.meta.cal_step.drop_integrations,
                         'SKIPPED')

    def test_regroup_unsupported_mode(self):
        """ Test the regroup step with unsupported mode. """
        test_model = datamodels.RampModel()
        test_model.meta.exposure.type = 'NIS_SOSS'
        di_ramp_model = DropIntegrationsStep().call(
            test_model, drop_integrations=[0])

        self.assertEqual(type(test_model), type(di_ramp_model))
        self.assertEqual(di_ramp_model.meta.cal_step.drop_integrations,
                         'SKIPPED')


if __name__ == '__main__':
    unittest.main()

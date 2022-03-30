import os
import unittest
import numpy as np

os.environ['CRDS_PATH'] = 'crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst import datamodels

from exotic_miri import Extract1dStep


class TestExtract1d(unittest.TestCase):
    """ Test extract 1d step. """

    def __init__(self, *args, **kwargs):
        super(TestExtract1d, self).__init__(*args, **kwargs)

        # Build test CubeModel data structure.
        self.test_cube_model = datamodels.CubeModel()
        self.test_cube_model.data = np.ones((100, 416, 72))
        self.test_cube_model.err = np.ones((100, 416, 72))
        self.test_cube_model.groupdq = np.zeros((100, 416, 72))
        self.test_cube_model.pixeldq = np.zeros((416, 72))
        self.test_cube_model.meta.ngroups = 50
        self.test_cube_model.meta.ngroups_file = 50
        self.test_cube_model.meta.nints = 1000
        self.test_cube_model.meta.nints_file = 1000
        self.test_cube_model.meta.exposure.integration_time = 10.3376
        self.test_cube_model.meta.exposure.ngroups = 50
        self.test_cube_model.meta.exposure.nints = 1000
        self.test_cube_model.meta.exposure.type = 'MIR_LRS-SLITLESS'
        self.test_cube_model.meta.exposure.readpatt = 'FASTR1'
        self.test_cube_model.meta.instrument.name = 'MIRI'
        self.test_cube_model.meta.instrument.detector = 'MIRIMAGE'
        self.test_cube_model.meta.observation.date = '2022-01-01'
        self.test_cube_model.meta.observation.time = '17:00:00'
        self.test_cube_model.meta.subarray.name = 'SLITLESSPRISM'
        self.test_cube_model.meta.subarray.xsize = 72
        self.test_cube_model.meta.subarray.xstart = 1
        self.test_cube_model.meta.subarray.ysize = 416
        self.test_cube_model.meta.subarray.ystart = 529
        self.test_cube_model.meta.bunit_data = 'DN/S'

    def test_extract_1d_bkg_const_spec_box(self):
        """ Test the extract 1d step: bkg=const, spec=box. """
        spectra_model = Extract1dStep().call(
            self.test_cube_model,
            bkg_algo='constant', bkg_region=[8, 22, 52, 70],
            extract_algo='box', extract_region_width=11)


if __name__ == '__main__':
    unittest.main()

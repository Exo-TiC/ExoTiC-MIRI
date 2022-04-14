import unittest
import numpy as np
from jwst import datamodels

from exotic_miri import ReferencePixelStep


class TestRefPixel(unittest.TestCase):
    """ Test reference pixel step. """

    def __init__(self, *args, **kwargs):
        super(TestRefPixel, self).__init__(*args, **kwargs)

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

    def test_ref_pixel_correction_sm_none_odd_even_false(self):
        """ Test the reference pixel step: smooth=none, odd/even=false. """
        ref_pix_ramp_model = ReferencePixelStep().call(
            self.test_ramp_model,
            smoothing_length=None,
            odd_even_rows=False)

        self.assertEqual(type(self.test_ramp_model),
                         type(ref_pix_ramp_model))
        self.assertEqual(ref_pix_ramp_model.data.shape,
                         (100, 50, 416, 72))

    def test_ref_pixel_correction_sm_various_odd_even_false(self):
        """ Test the reference pixel step: smooth=various, odd/even=false. """
        smooth_lengths = [1, 2, 5, 11, 101]
        for sl in smooth_lengths:
            ref_pix_ramp_model = ReferencePixelStep().call(
                self.test_ramp_model,
                smoothing_length=sl,
                odd_even_rows=False)

            self.assertEqual(type(self.test_ramp_model),
                             type(ref_pix_ramp_model))
            self.assertEqual(ref_pix_ramp_model.data.shape,
                             (100, 50, 416, 72))

    def test_ref_pixel_correction_sm_none_odd_even_true(self):
        """ Test the reference pixel step: smooth=none, odd/even=true. """
        ref_pix_ramp_model = ReferencePixelStep().call(
            self.test_ramp_model,
            smoothing_length=None,
            odd_even_rows=True)

        self.assertEqual(type(self.test_ramp_model),
                         type(ref_pix_ramp_model))
        self.assertEqual(ref_pix_ramp_model.data.shape,
                         (100, 50, 416, 72))

    def test_ref_pixel_correction_sm_various_odd_even_true(self):
        """ Test the reference pixel step: smooth=various, odd/even=true. """
        smooth_lengths = [1, 2, 5, 11, 101]
        for sl in smooth_lengths:
            ref_pix_ramp_model = ReferencePixelStep().call(
                self.test_ramp_model,
                smoothing_length=sl,
                odd_even_rows=True)

            self.assertEqual(type(self.test_ramp_model),
                             type(ref_pix_ramp_model))
            self.assertEqual(ref_pix_ramp_model.data.shape,
                             (100, 50, 416, 72))

    def test_ref_pix_incorrect_data_model(self):
        """ Test the ref pix step with incorrect data model. """
        test_model = datamodels.CubeModel()
        test_model.meta.exposure.type = 'MIR_LRS-SLITLESS'
        ref_pix_ramp_model = ReferencePixelStep().call(
            test_model,
            smoothing_length=None,
            odd_even_rows=False)

        self.assertEqual(type(test_model), type(ref_pix_ramp_model))
        self.assertEqual(ref_pix_ramp_model.meta.cal_step.refpix,
                         'SKIPPED')

    def test_ref_pix_unsupported_mode(self):
        """ Test the ref pix step with unsupported mode. """
        test_model = datamodels.RampModel()
        test_model.meta.exposure.type = 'NIS_SOSS'
        ref_pix_ramp_model = ReferencePixelStep().call(
            test_model,
            smoothing_length=None,
            odd_even_rows=False)

        self.assertEqual(type(test_model), type(ref_pix_ramp_model))
        self.assertEqual(ref_pix_ramp_model.meta.cal_step.refpix,
                         'SKIPPED')


if __name__ == '__main__':
    unittest.main()

import os
import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step


class IntegrationTimesStep(Step):
    """ Get integration times step.
    This steps enables the user to get and save the integration times.
    """

    spec = """
    data_chunk_name = string(default=None)  # data base name.
    stage_1_dir = string(default=None)  # directory of stage 1 products.
    stage_2_dir = string(default=None)  # directory of stage 2 products.
    """

    def process(self, input):
        """Execute the step.
        Parameters
        ----------
        input: JWST data model
            A data model of type CubeModel.
        Returns
        -------
        array and float
            Array of integration times (BJD TDB) and duration of an
            integration in seconds.
        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error('Input is a {} which was not expected for '
                               'IntegrationTimesStep, skipping step.'.format(
                                str(type(input_model))))
                return input_model

            # Concat times from all chunks.
            mid_int_times = np.array(input_model.int_times['int_mid_BJD_TDB'])

        int_time_s = np.median(np.diff(mid_int_times)) * 24. * 3600.
        self.log.info('Integration duration={} secs'.format(int_time_s))

        # Save.
        hdu = fits.PrimaryHDU(mid_int_times)
        hdul = fits.HDUList([hdu])
        integrations_name = '{}_stage_2_integration_times.fits'.format(
            self.data_chunk_name)
        hdul.writeto(os.path.join(
            self.stage_2_dir, integrations_name), overwrite=True)

        return mid_int_times, int_time_s

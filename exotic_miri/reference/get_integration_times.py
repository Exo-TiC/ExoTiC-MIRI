import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class GetIntegrationTimes(Step):
    """ Get the integration times. """

    spec = """
    
    """

    def process(self, input):
        """ Get and save the integration times for a data segment, and
        compute the integration duration in seconds.

        Parameters
        ----------
        input: jwst.datamodels.RampModel
            This is an uncal.fits loaded data segment.

        Returns
        -------
        timing_data and integration_duration: tuple(fits.table, float)
            Table of integration times data and duration of an
            integration in seconds.

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel):
                self.log.error("Input is a {} which was not expected for "
                               "IntegrationTimesStep, skipping step.".format(
                                str(type(input_model))))
                return input_model

            # Extract data.
            timing_data = input_model.int_times
            int_time_s = np.median(np.diff(
                input_model.int_times["int_mid_BJD_TDB"])) * 24. * 3600.
            self.log.info("Integration duration={} secs".format(int_time_s))

        return timing_data, int_time_s

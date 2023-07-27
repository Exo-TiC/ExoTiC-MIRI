import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class GetIntegrationTimes(Step):
    """ Get the integration times.

    This enables the user to get and save the integration times, and
    computes the integration duration in seconds.

    """

    spec = """
    
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type CubeModel.

        Returns
        -------
        (fits.table, float)
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

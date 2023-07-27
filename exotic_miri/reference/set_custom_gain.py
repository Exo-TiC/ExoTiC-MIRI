import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class SetCustomGain(Step):
    """ Set a custom gain.

    This enables the user to set a custom gain value.

    """

    spec = """
    gain_value = float(default=None)  # constant gain value.
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type CubeModel.

        Returns
        -------
        JWST gain_model
            A gain model for override_gain usage.

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel):
                self.log.error("Input is a {} which was not expected for "
                               "CustomGainStep, skipping step.".format(
                                str(type(input_model))))
                return None

            # Use default reference file as template for custom file.
            self.log.info("Building custom gain datamodel.")
            gain_ref_name = self.get_reference_file(input_model, "gain")
            gain_model = datamodels.GainModel(gain_ref_name)
            if self.gain_value is not None:
                gain_model.data = np.ones_like(gain_model.data) * self.gain_value

        return gain_model

    def finalize_result(self, res, ref):
        """
        :meta private:
        """
        # Required to enable ref model to be returned.
        # Overwrites base class method.
        pass

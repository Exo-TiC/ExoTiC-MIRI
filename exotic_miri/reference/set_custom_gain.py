import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class SetCustomGain(Step):
    """ Set a custom gain. """

    spec = """
    gain_value = float(default=3.1)  # constant gain value.
    """

    def process(self, input):
        """ Set a custom gain value. This step creates a gain model
        which can be passed to other steps for processing. It should
        be passed to all steps that make use of the gain, and passed
        via the arg 'override_gain'. This includes stage 1 steps such
        as jwst.calwebb_detector1.jump_step,
        jwst.calwebb_detector1.ramp_fit_step, and
        jwst.calwebb_detector1.gain_scale_step.

        Parameters
        ----------
        input: jwst.datamodels.RampModel
            This is an uncal.fits loaded data segment.
        gain_value: float
            The gain value to set. Default is 3.1.

        Returns
        -------
        gain : jwst.datamodels.GainModel
            The gain model which can be passed to other steps.

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

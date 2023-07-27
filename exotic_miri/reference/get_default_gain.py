import os
import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step


class GetDefaultGain(Step):
    """ Get the default gain.

    This enables the user to get and save gain data from
    the default CRDS files.

    """

    spec = """
    median_value = boolean(default=False)  # only return median value.
    save = boolean(default=False)  # save gain model to disk as .fits.
    save_path = boolean(default=False)  # save gain model path.
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type CubeModel.

        Returns
        -------
        GainModel or float
            Gain model, or float if median_value=True.

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel) \
                    and not isinstance(input_model, datamodels.CubeModel):
                self.log.error("Input is a {} which was not expected for "
                               "GainStep, skipping step.".format(
                                str(type(input_model))))
                return input_model

            # Extract default gain model.
            self.log.info("Getting default gain model.")
            gain_filename = self.get_reference_file(input_model, "gain")
            gain_model = datamodels.GainModel(gain_filename)

        if self.save:
            gain_model.save(path=os.path.join(
                self.save_path, "default_gain.fits"))

        # Median value.
        med_gain = np.median(gain_model.data)
        self.log.info("Median gain={} electrons/DN.".format(med_gain))

        if self.median_value:
            return float(med_gain)
        else:
            return gain_model

    def finalize_result(self, res, ref):
        # Required to enable ref model to be returned.
        # Overwrites base class method.
        pass

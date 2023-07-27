import os
import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step


class GetDefaultReadNoise(Step):
    """ Get the default readnoise.

    This enables the user to get and save readnoise data from
    the default CRDS files.

    """

    spec = """
    median_value = boolean(default=False)  # only return median value.
    save = boolean(default=False)  # save readnoise model to disk as .fits.
    save_path = boolean(default=False)  # save readnoise model path.
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type CubeModel.

        Returns
        -------
        ReadnoiseModel or float
            Readnoise model, or float if median_value=True.

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel) \
                    and not isinstance(input_model, datamodels.CubeModel):
                self.log.error("Input is a {} which was not expected for "
                               "ReadNoiseStep, skipping step.".format(
                                str(type(input_model))))
                return input_model

            # Extract default readnoise model.
            self.log.info("Getting default readnoise model.")
            readnoise_filename = self.get_reference_file(input_model, "readnoise")
            readnoise_model = datamodels.ReadnoiseModel(readnoise_filename)

        if self.save:
            readnoise_model.save(path=os.path.join(
                self.save_path, "default_readnoise.fits"))

        # Median value.
        med_readnoise = np.median(readnoise_model.data)
        self.log.info("Median readnoise={} DN.".format(med_readnoise))

        if self.median_value:
            return float(med_readnoise)
        else:
            return readnoise_model

    def finalize_result(self, res, ref):
        """
        :meta private:
        """
        # Required to enable ref model to be returned.
        # Overwrites base class method.
        pass

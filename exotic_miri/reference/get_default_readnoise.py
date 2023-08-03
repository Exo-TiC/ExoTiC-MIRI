import os
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class GetDefaultReadNoise(Step):
    """ Get the default readnoise. """

    spec = """
    median_value = boolean(default=False)  # only return median value.
    save = boolean(default=False)  # save readnoise model to disk as .fits.
    save_path = string(default=None)  # save readnoise model path.
    """

    def process(self, input):
        """ Get and save readnoise data from the default CRDS files.

        Parameters
        ----------
        input: jwst.datamodels.RampModel or jwst.datamodels.CubeModel
            This is either an uncal.fits or rateints.fits loaded
            data segment. The readnoise will be the same no matter which
            data segment you pass in.
        median_value : boolean
            If True only return the median value rather than the readnoise
            model. Default is False.
        save : boolean
            If True save the readnoise model to disc. Default is False.
        save_path : string
            If save==True save the readnoise model to this path. Default
            is None.

        Returns
        -------
        if median_value == False:
            gain : jwst.datamodels.ReadnoiseModel
                The readnoise model which can be passed to other steps.
        elif median_value == True:
            gain : float
                The median readnoise value on the entire detector.

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

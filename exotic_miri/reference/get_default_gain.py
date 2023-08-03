import os
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class GetDefaultGain(Step):
    """ Get the default gain. """

    spec = """
    median_value = boolean(default=False)  # only return median value.
    save = boolean(default=False)  # save gain model to disk as .fits.
    save_path = string(default=None)  # save gain model path.
    """

    def process(self, input):
        """ Get and save gain data from the default CRDS files.

        Parameters
        ----------
        input: jwst.datamodels.RampModel or jwst.datamodels.CubeModel
            This is either an uncal.fits or rateints.fits loaded
            data segment. The gain will be the same no matter which
            data segment you pass in.
        median_value : boolean
            If True only return the median value rather than the gain
            model. Default is False.
        save : boolean
            If True save the gain model to disc. Default is False.
        save_path : string
            If save==True save the gain model to this path. Default
            is None.

        Returns
        -------
        if median_value == False:
            gain : jwst.datamodels.GainModel
                The gain model which can be passed to other steps.
        elif median_value == True:
            gain : float
                The median gain value on the entire detector.

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
        """
        :meta private:
        """
        # Required to enable ref model to be returned.
        # Overwrites base class method.
        pass

from jwst import datamodels
from jwst.stpipe import Step


class ReferencePixelStep(Step):
    """ Reference pixel correction.

    This steps allows the user to apply corrections to their group-
    level images, using the reference pixels, for the MIRI LRS
    subarray. The corrections can be made with a variety of options
    for smoothing the values and/or separating odd and even rows.

    """

    spec = """
    smoothing_length = integer(default=None)  # median smooth values over pixel length
    odd_even_rows = boolean(default=True)  # treat and odd and even rows separately
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type RampModel.

        Returns
        -------
        JWST data model
            A RampModel with updated integration groupings, unless the
            step is skipped in which case `input_model` is returned.
        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            rpc_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel):
                self.log.error('Input is a {} which was not expected for '
                               'reference_pixel_step, skipping step.'
                               .format(str(type(input_model))))
                rpc_model.meta.cal_step.refpix = 'SKIPPED'
                return rpc_model

            # Check the observation mode.
            if not input_model.meta.exposure.type == 'MIR_LRS-SLITLESS':
                self.log.error('Observation is a {} which is not supported '
                               'by ExoTic-MIRIs reference_pixel_step, '
                               'skipping step.'.format(
                                input_model.meta.exposure.type))
                rpc_model.meta.cal_step.refpix = 'SKIPPED'
                return rpc_model

            # Do reference pixel correction.
            # Iterate integrations.
            # Subtract first group from all groups.
            # Compute the corrections.
            ref_pix_corr = self.compute_reference_pixel_correction()
            # Subtract the corrections.
            # Add first group back to all groups.

            # Update meta.
            rpc_model.meta.cal_step.refpix = 'COMPLETE'

        return rpc_model

    def compute_reference_pixel_correction(self):
        """ Compute the reference pixel correction. """
        # Compute per amplifier per odd/even rows and smooth ref pix values.
        # Do not include pixels with flags.
        # Iteratively sigma clip these values perhaps.
        # Return the correction grid for the integration.
        return

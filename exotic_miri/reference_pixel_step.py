import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class ReferencePixelStep(Step):
    """ Reference pixel correction.

    This steps enables the user to apply corrections to their group-
    level images, using the reference pixels available to the MIRI LRS
    subarray. The corrections can be made with a variety of options
    for smoothing the values and/or separating odd and even rows.

    """

    spec = """
    smoothing_length = integer(default=None)  # median smooth values over pixel length
    odd_even_rows = boolean(default=True)  # treat odd and even rows separately
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
            A RampModel with the reference pixel correction applied, unless
            the step is skipped in which case `input_model` is returned.
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

            # Iterate integrations.
            for idx_int, integration in enumerate(rpc_model.data):

                # Subtract first group from all groups.
                int_ims = integration - integration[0]

                # Compute and subtract the corrections.
                # todo: mask/replace flagged ref pixels.
                int_ims -= self.compute_reference_pixel_correction(int_ims)

                # Add first group back to all groups.
                rpc_model.data[idx_int] = int_ims + integration[0]

            # Update meta.
            rpc_model.meta.cal_step.refpix = 'COMPLETE'

        return rpc_model

    def compute_reference_pixel_correction(self, int_ims):
        """ Compute the reference pixel correction tiles. """
        ref_pixels = int_ims[:, :, 0:4]
        if not self.odd_even_rows:
            if self.smoothing_length is None:
                return np.tile(
                    np.median(ref_pixels, axis=1)[:, np.newaxis, :],
                    (1, int_ims.shape[1], int(int_ims.shape[2] / 4)))
            else:
                return np.tile(
                    self._median_smooth_per_column(ref_pixels),
                    (1, 1, int(int_ims.shape[2] / 4)))
        else:
            odd_even_medians = np.copy(ref_pixels)
            if self.smoothing_length is None:
                odd_even_medians[:, 0::2, :] = np.median(
                    odd_even_medians[:, 0::2, :], axis=1)[:, np.newaxis, :]
                odd_even_medians[:, 1::2, :] = np.median(
                    odd_even_medians[:, 1::2, :], axis=1)[:, np.newaxis, :]
                return np.tile(
                    odd_even_medians, (1, 1, int(int_ims.shape[2] / 4)))
            else:
                odd_even_medians[:, 0::2, :] = self._median_smooth_per_column(
                    odd_even_medians[:, 0::2, :])
                odd_even_medians[:, 1::2, :] = self._median_smooth_per_column(
                    odd_even_medians[:, 1::2, :])
                return np.tile(
                    odd_even_medians, (1, 1, int(int_ims.shape[2] / 4)))

    def _median_smooth_per_column(self, ref_pixels):
        """ Median smooth data per column. """
        sm_ref_pixels = np.copy(ref_pixels)
        n_rows = ref_pixels.shape[1]
        sm_radius = int((self.smoothing_length - 1) / 2)
        for idx_row in range(n_rows):
            # Define window.
            start_row = max(0, idx_row - sm_radius)
            end_row = min(n_rows - 1, idx_row + sm_radius)

            # Compute median in window.
            sm_ref_pixels[:, idx_row, :] = np.median(
                ref_pixels[:, start_row:end_row + 1, :], axis=1)

        return sm_ref_pixels

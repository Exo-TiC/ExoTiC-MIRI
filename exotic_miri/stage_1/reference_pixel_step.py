import matplotlib.pyplot as plt
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class ReferencePixelStep(Step):
    """ Reference pixel step. """

    spec = """
    smoothing_length = integer(default=None)  # median smooth values over pixel length
    odd_even_rows = boolean(default=True)  # treat odd and even rows separately
    draw_correction = boolean(default=False)  # draw correction images
    """

    def process(self, input):
        """ Reference pixel correction at the group level, using the
        reference pixels available to the MIRI LRS subarray. The corrections
        can be made with a variety of options for smoothing the values and/or
        separating odd and even rows.

        The default pipeline, at the time of programming, does not have
        this option for subarrays. It assumes subarrays have no available
        reference pixels, but the MIRI LRS subarray is against the edge of
        the detector.

        Parameters
        ----------
        input: jwst.datamodels.RampModel
            This is an uncal.fits loaded data segment.
        smoothing_length: integer
            If not None, the number of rows to median smooth the estimated
            reference pixel values over. Default is no smoothing.
        odd_even_rows: boolean
            Treat the correction separately for odd and even rows. Default
            is True.
        draw_correction: boolean
            Plot the correction images.

        Returns
        -------
        output: jwst.datamodels.RampModel
            A RampModel with the reference pixel correction applied.

        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            rpc_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel):
                self.log.error("Input is a {} which was not expected for "
                               "reference_pixel_step, skipping step."
                               .format(str(type(input_model))))
                rpc_model.meta.cal_step.refpix = "SKIPPED"
                return rpc_model

            # Check the observation mode.
            if not input_model.meta.exposure.type == "MIR_LRS-SLITLESS":
                self.log.error("Observation is a {} which is not supported "
                               "by ExoTic-MIRIs reference_pixel_step, "
                               "skipping step.".format(
                                input_model.meta.exposure.type))
                rpc_model.meta.cal_step.refpix = "SKIPPED"
                return rpc_model

            # Iterate integrations.
            for idx_int, integration in enumerate(rpc_model.data):

                # Subtract first group from all groups.
                int_ims = integration - integration[0]

                # Compute and subtract the corrections.
                # todo: mask flagged ref pixels.
                ref_pixel_correction = self._compute_reference_pixel_correction(int_ims)

                if self.draw_correction:
                    self._draw_correction_images(idx_int, ref_pixel_correction)

                int_ims -= ref_pixel_correction

                # Add first group back to all groups.
                rpc_model.data[idx_int] = int_ims + integration[0]

            # Update meta.
            rpc_model.meta.cal_step.refpix = "COMPLETE"

        return rpc_model

    def _compute_reference_pixel_correction(self, int_ims):
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

    def _draw_correction_images(self, idx_int, ref_pixel_correction):
        for idx_grp, grp_correction in enumerate(ref_pixel_correction):
            fig, ax1 = plt.subplots(1, 1, figsize=(5, 7))
            ax1.imshow(grp_correction, origin="lower")
            ax1.set_title("Integration={}, group={}".format(idx_int, idx_grp))
            ax1.set_xlabel("Column pixels")
            ax1.set_ylabel("Row pixels")
            plt.tight_layout()
            plt.show()

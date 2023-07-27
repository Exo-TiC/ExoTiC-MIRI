import numpy as np
from jwst import datamodels
from jwst.stpipe import Step
import matplotlib.pyplot as plt


class BackgroundSubtractStep(Step):
    """ Background subtraction step.
    This steps enables the user to subtract the background.
    """

    spec = """
    method = string(default="row_wise")  # background subtraction method: constant, row_wise, col_wise.
    bkg_col_left_start = integer(default=8)  # left-side background start.
    bkg_col_left_end = integer(default=17)  # left-side background end.
    bkg_col_right_start = integer(default=56)  # right-side background start.
    bkg_col_right_end = integer(default=72)  # right-side background end.
    smoothing_length = integer(default=None)  # median smooth values over pixel length
    """

    def process(self, input):
        """Execute the step.
        Parameters
        ----------
        input: JWST data model
            A data model of type CubeModel.
        Returns
        -------
        JWST data model and background cube
            A CubeModel with background subtracted, and a 3d np.array of the
            estimated background.
        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            bkg_subtracted_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error("Input is a {} which was not expected for "
                               "BackgroundSubtractStep, skipping step.".format(
                                str(type(input_model))))
                return input_model

            if self.method == "constant":
                bkg = self.constant_background(input_model.data)
            elif self.method == "row_wise":
                bkg = self.row_wise_background(input_model.data)
            elif self.method == "col_wise":
                bkg = self.col_wise_background(input_model.data)
            else:
                raise ValueError("Background method not recognised.")

        bkg_subtracted_model.data -= bkg

        return bkg_subtracted_model, bkg

    def constant_background(self, data):
        """ One value per integration. """
        bkd_col_idxs = np.concatenate(
            [np.arange(self.bkg_col_left_start, self.bkg_col_left_end),
             np.arange(self.bkg_col_right_start, self.bkg_col_right_end)])
        bkg = np.median(data[:, :, bkd_col_idxs], axis=(1, 2))
        return np.tile(bkg[:, np.newaxis, np.newaxis],
                       (1, data.shape[1], data.shape[2]))

    def row_wise_background(self, data):
        """ One value per row per integration. """
        bkd_col_idxs = np.concatenate(
            [np.arange(self.bkg_col_left_start, self.bkg_col_left_end),
             np.arange(self.bkg_col_right_start, self.bkg_col_right_end)])
        bkg = np.median(data[:, :, bkd_col_idxs], axis=2)
        bkg = np.tile(bkg[:, :, np.newaxis], (1, 1, data.shape[2]))
        if self.smoothing_length is None:
            return bkg
        else:
            return self._median_smooth_per_column(bkg)

    def col_wise_background(self, data):
        """ One value per row per column per integration. """
        # ll = 12
        # lr = 36
        # rl = 36
        # rr = 72
        #
        # bkg_left = np.median(data[:, :, ll:lr], axis=(0, 1))
        # bkg_right = np.median(data[:, :, rl:rr], axis=(0, 1))
        # plt.scatter(np.arange(ll, lr), bkg_left)
        # plt.scatter(np.arange(rl, rr), bkg_right)
        # plt.show()
        #
        # bkg_left = np.median(data[:, 0::2, ll:lr], axis=(0, 1))
        # bkg_right = np.median(data[:, 0::2, rl:rr], axis=(0, 1))
        # plt.scatter(np.arange(ll, lr), bkg_left)
        # plt.scatter(np.arange(rl, rr), bkg_right)
        # bkg_left = np.median(data[:, 1::2, ll:lr], axis=(0, 1))
        # bkg_right = np.median(data[:, 1::2, rl:rr], axis=(0, 1))
        # plt.scatter(np.arange(ll, lr), bkg_left)
        # plt.scatter(np.arange(rl, rr), bkg_right)
        # plt.show()
        #
        # for row_idx in range(150, 400, 40):
        #     bkg_left = np.median(data[:, row_idx:row_idx+20, ll:lr], axis=(0, 1))
        #     bkg_right = np.median(data[:, row_idx:row_idx+20, rl:rr], axis=(0, 1))
        #     plt.scatter(np.arange(ll, lr), bkg_left)
        #     plt.scatter(np.arange(rl, rr), bkg_right)
        #     plt.show()
        #
        # for row_idx in range(150, 400, 40):
        #     bkg_left = np.median(data[:, row_idx:row_idx+20, ll:lr], axis=(0, 1))
        #     bkg_right = np.median(data[:, row_idx:row_idx+20, rl:rr], axis=(0, 1))
        #     plt.scatter(np.concatenate([np.arange(ll, lr), np.arange(rl, rr)]),
        #                 np.concatenate([bkg_left, bkg_right]))
        # plt.show()

        col_pixel_idxs = np.tile(np.arange(data.shape[2])[:, np.newaxis],
                                 (1, data.shape[1]))
        ls_mid_col = (self.bkg_col_left_start + self.bkg_col_left_end) / 2
        rs_mid_col = (self.bkg_col_right_start + self.bkg_col_right_end) / 2
        xs = np.array([ls_mid_col, rs_mid_col])

        ls_mid_val = np.median(
            data[:, :, self.bkg_col_left_start:self.bkg_col_left_end], axis=2)
        rs_mid_val = np.median(
            data[:, :, self.bkg_col_right_start:self.bkg_col_right_end], axis=2)
        ys = np.concatenate(
            [ls_mid_val[:, np.newaxis, :], rs_mid_val[:, np.newaxis, :]], axis=1)

        bkg = np.empty(data.shape)
        for int_idx in range(data.shape[0]):
            p_coeff = np.polyfit(xs, ys[int_idx], deg=1)
            bkg[int_idx] = np.polyval(p_coeff, col_pixel_idxs).T

        if self.smoothing_length is None:
            return bkg
        else:
            return self._median_smooth_per_column(bkg)

    def _median_smooth_per_column(self, col_pixels):
        """ Median smooth data per column. """
        sm_col_pixels = np.copy(col_pixels)
        n_rows = col_pixels.shape[1]
        sm_radius = int((self.smoothing_length - 1) / 2)
        for idx_row in range(n_rows):
            # Define window.
            start_row = max(0, idx_row - sm_radius)
            end_row = min(n_rows - 1, idx_row + sm_radius)

            # Compute median in window.
            sm_col_pixels[:, idx_row, :] = np.median(
                col_pixels[:, start_row:end_row + 1, :], axis=1)

        return sm_col_pixels

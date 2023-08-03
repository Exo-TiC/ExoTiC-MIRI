import warnings
import numpy as np
from jwst import datamodels
from jwst.stpipe import Step
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CleanOutliersStep(Step):
    """ Clean outliers step. """

    spec = """
    window_heights = int_list(default=None)  # window heights for spatial profile fitting.
    dq_bits_to_mask = int_list(default=None)  # dq flags for which pixels to clean.
    poly_order = integer(default=4)  # spatial profile polynomial fitting order.
    outlier_threshold = float(default=4.0)  # spatial profile fitting outlier sigma.
    spatial_profile_left_idx = integer(default=26)  # left-side of aperture width.
    spatial_profile_right_idx = integer(default=47)  # right-side of aperture width.
    draw_cleaning_col = boolean(default=False)  # draw columns of window cleaning.
    draw_spatial_profiles = boolean(default=False)  # draw spatial profile.
    no_clean = boolean(default=False)  # skip clean, just rm nans, for quick tests.
    """

    def __int__(self):
        self.D = None
        self.S = None
        self.V = None
        self.V0 = None
        self.G = None
        self.P = None
        self.DQ_pipe = None
        self.DQ_spatial = None

    def process(self, input):
        """ Clean outliers using deviations from a spatial profile, following
        Horne 1986, and/or values from the data quality array.

        NB. DQ array bit values as per:
        `https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html?#data-quality-flags`.

        Parameters
        ----------
        input: jwst.datamodels.CubeModel
            This is a rateints.fits loaded data segment.
        window_heights: list of integers
            The size of the windows in pixels, in the dispersion direction, to
            use when fitting polynomials to the spatial profile. The size of the
            window iterates cyclically through the list until the total height
            of the detector is reached. Recommended to use smaller window sizes
            at the shorter wavelengths (larger row indexes) as the throughput *
            stellar spectra show larger variations here. For example,
            [150, 100, 50, 50, 20, 20, 20].
        dq_bits_to_mask: list of integers
            A list of data quality flags to clean. These pixels are replaced by
            the spatial profile values. See link above for definitions of the
            DQ bit values. For example, [0, ] cleans pixes marked as 2**0
            (do_not_use) in the DQ array.
        poly_order: integer
            Polynomial order for fitting to the windows of data. Default is 4.
        outlier_threshold: float
            Number of standard deviations away from the spatial profile for a
            pixel to be determined as an outlier. Default is 4.0.
        spatial_profile_left_idx: integer
            Start index of the columns which should be included in the spatial
            profile. Default is 26.
        spatial_profile_right_idx: integer
            End index of the columns which should be included in the spatial
            profile. Default is 47.
        draw_cleaning_col: boolean
            Plot the cleaning interactively. Useful for understanding the process
            and getting a feel for the hyperparams.
        draw_spatial_profiles: boolean
            Plot the spatial profiles after each integration is cleaned.
        no_clean: boolean
            Override, and just remove any nans. This is for quick tests.

        Returns
        -------
        output, spatial profile cube, outlier counts cube: tuple(CubeModel, np.ndarray, np.ndarray)
            A CubeModel with outliers cleaned, a 3D array of the
            fitted spatial profiles, and a count of the number of outliers
            cleaned within 0-4 pixels of the spectral trace (column index 36).

        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            cleaned_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error("Input is a {} which was not expected for "
                               "CleanOutliersStep, skipping step.".format(
                                str(type(input_model))))
                return input_model

            if self.no_clean:
                cleaned_model.data = np.nan_to_num(cleaned_model.data)
                outliers = self._count_outliers(cleaned_model)
                return cleaned_model, np.empty(cleaned_model.shape), outliers

            self.D = input_model.data
            self.V = input_model.err**2
            self.P = np.empty(self.D.shape)
            self.DQ_pipe = input_model.dq
            self.DQ_spatial = np.ones(self.D.shape).astype(bool)

            # Clean via col-wise windowed spatial profile fitting.
            self._clean()

        cleaned_model.data = self.D
        cleaned_model.dq += (~self.DQ_spatial).astype(np.uint32) * 2**4  # Set as outlier.

        # Count number of replaced pixels on and near the spectral trace.
        outliers = self._count_outliers(cleaned_model)

        return cleaned_model, self.P, outliers

    def _clean(self):
        """ Clean dq bits and via optimal extraction method of Horne 1986. """
        # Prep cycling of window heights to span all rows.
        n_ints, n_rows, n_cols = self.D.shape
        window_heights = np.tile(self.window_heights, int(np.ceil(
            n_rows / np.sum(self.window_heights))))
        window_start_idxs = np.concatenate([[0, ], np.cumsum(window_heights)])

        # Iterate integrations.
        for int_idx in range(n_ints):

            # Iterate windows of several rows.
            for win_start_idx, win_end_idx in zip(window_start_idxs, window_start_idxs[1:]):

                # Set window in rows.
                win_start_idx = min(win_start_idx, n_rows)
                win_end_idx = min(win_end_idx, n_rows)
                win_width = win_end_idx - win_start_idx
                if win_width == 0:
                    continue

                # Spatial profile cleaning, and dq bit cleaning too.
                P_win = self._spatial_profile_cleaning(
                    int_idx, win_start_idx, win_end_idx, win_width)

                # Normalise.
                P_win_sub = P_win[:, self.spatial_profile_left_idx:self.spatial_profile_right_idx]
                norm_win = np.sum(P_win_sub, axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    P_win_sub /= norm_win[:, np.newaxis]
                if np.isnan(P_win_sub).any():
                    # self.log.warn(
                    #     "Spatial profile contains entire slice of negative "
                    #     "values. Setting as top hat.")
                    n_rows_win, n_cols_win = P_win_sub.shape
                    for row_win_idx, n in enumerate(norm_win):
                        if n == 0:
                            P_win_sub[row_win_idx, :] = 1. / n_rows_win
                P_win_full = np.concatenate(
                    [np.zeros((P_win.shape[0], self.spatial_profile_left_idx)),
                     P_win_sub,
                     np.zeros((P_win.shape[0], P_win.shape[1] - self.spatial_profile_right_idx))], axis=1)

                # Save window of spatial profile.
                self.P[int_idx, win_start_idx:win_end_idx, :] = P_win_full

            if self.draw_spatial_profiles:
                self._draw_spatial_profile(int_idx)

            self.log.info("Integration={}: cleaned {} outliers "
                          "w/ spatial profile.".format(
                           int_idx, np.sum(~self.DQ_spatial[int_idx])))

    def _spatial_profile_cleaning(self, int_idx, win_start_idx, win_end_idx, win_width):
        """ P as per Horne 1986 table 1 (step 5). """
        P = []
        D_S = self.D[int_idx, win_start_idx:win_end_idx, :]
        DQ_pipe_S = self.DQ_pipe[int_idx, win_start_idx:win_end_idx, :]

        # Iterate cols in window.
        row_pixel_idxs = np.arange(D_S.shape[0])
        for col_idx in np.arange(0, D_S.shape[1]):

            D_S_col = np.copy(D_S[:, col_idx])
            col_mask = np.ones(D_S_col.shape[0]).astype(bool)

            # Set nans as bad.
            col_mask[~np.isfinite(D_S_col)] = False

            # Set selected dq flags as bad.
            for win_idx, pixel_binary_sum in enumerate(DQ_pipe_S[:, col_idx]):
                bit_array = np.flip(list(np.binary_repr(pixel_binary_sum, width=32))).astype(bool)
                if np.any(bit_array[self.dq_bits_to_mask]):
                    col_mask[win_idx] = False

            while True:
                try:
                    if np.sum(col_mask) < 2:
                        raise TypeError
                    # Fit polynomial to row.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.RankWarning)
                        p_coeff = np.polyfit(
                            row_pixel_idxs[col_mask], D_S_col[col_mask],
                            self.poly_order, w=None)
                        p_row = np.polyval(p_coeff, row_pixel_idxs)

                except (np.linalg.LinAlgError, TypeError) as err:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.RankWarning)
                        p_coeff = np.polyfit(
                            row_pixel_idxs, np.zeros(D_S_col.shape[0]),
                            self.poly_order, w=None)
                        p_row = np.polyval(p_coeff, row_pixel_idxs)

                # Check residuals to polynomial fit.
                res_col = np.ma.array(D_S_col - p_row, mask=~col_mask)
                dev_col = np.ma.abs(res_col) / np.ma.std(res_col)
                max_deviation_idx = np.ma.argmax(dev_col)
                if dev_col[max_deviation_idx] > self.outlier_threshold:
                    # Outlier: mask and repeat poly fitting.
                    if self.draw_cleaning_col:
                        print("Max dev={} > threshold={}".format(
                            dev_col[max_deviation_idx], self.outlier_threshold))
                        self._draw_poly_inter_fit(
                            int_idx, col_idx, win_start_idx, win_end_idx,
                            win_width, p_row, col_mask)

                    col_mask[max_deviation_idx] = False
                    self.DQ_spatial[int_idx, win_start_idx + max_deviation_idx,
                                    col_idx] = False
                    continue
                else:
                    P.append(p_row)

                    # Replace data with poly val.
                    for win_idx, good_pix in enumerate(col_mask):
                        if not good_pix:
                            self.D[int_idx, win_start_idx + win_idx,
                                   col_idx] = np.polyval(
                                       p_coeff, win_idx)

                            # Set for nans.
                            self.DQ_spatial[int_idx, win_start_idx + win_idx,
                                            col_idx] = False

                    if self.draw_cleaning_col:
                        print("Max dev={} > threshold={}".format(
                            dev_col[max_deviation_idx], self.outlier_threshold))
                        print("Final cleaned data and fit.")
                        self._draw_poly_inter_fit(
                            int_idx, col_idx, win_start_idx, win_end_idx,
                            win_width, p_row, final=True)
                    break

        # Enforce positivity.
        P = np.array(P).T
        P[P < 0.] = 0.

        return P

    def _count_outliers(self, cleaned_model):
        outliers = []
        for region_width in range(5):
            outliers.append(np.sum(cleaned_model.dq[
                :, :, 36 - region_width:36 + region_width + 1] > 0,
                axis=2)[:, :, np.newaxis])

        return np.concatenate(outliers, axis=2)

    def _draw_poly_fit(self, idx_slice, x_data, y_data, x_model, y_model):
        """ Draw the polynomial fit. """
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))
        ax1.scatter(x_data, y_data, s=10, c="#000000", alpha=0.8,
                    label="Data, slice={}.".format(idx_slice))
        ax1.plot(x_model, y_model, c="#bc5090", lw=3,
                 label="Poly fit, order={}.".format(self.poly_order))
        ax1.set_xlabel("Pixel")
        ax1.set_ylabel("Electrons")
        plt.legend(loc="upper center")
        plt.tight_layout()
        plt.show()

    def _draw_poly_inter_fit(self, int_idx, col_idx, win_start_idx, win_end_idx,
                            win_width, p_col, col_mask=None, final=False):
        """ Draw the polynomial fit. """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 8))
        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.get_shared_y_axes().join(ax1, ax2)

        ax1.imshow(~self.DQ_spatial[int_idx], origin="lower",
                   aspect="auto", interpolation="none")
        im = self.D[int_idx]
        ax2.imshow(im, origin="lower", aspect="auto",
                   interpolation="none",
                   vmin=np.nanpercentile(im, 0.5),
                   vmax=np.nanpercentile(im, 99.5))

        rect = patches.Rectangle(
            (col_idx - 0.5, win_start_idx - 0.5), 1, win_width,
            linewidth=1, edgecolor="#ffffff", facecolor="none")
        ax1.add_patch(rect)

        if not final:
            x = np.arange(win_start_idx, win_end_idx)
            y = self.D[int_idx, win_start_idx:win_end_idx, col_idx]
            ax3.scatter(y[col_mask], x[col_mask], s=10, c="#000000",
                        alpha=0.8, label="Col={}.".format(col_idx))
            ax3.plot(p_col, x, c="#bc5090", lw=3,
                     label="Poly fit, order={}.".format(self.poly_order))
        else:
            x = np.arange(win_start_idx, win_end_idx)
            y = self.D[int_idx, win_start_idx:win_end_idx, col_idx]
            ax3.scatter(y, x, s=10, c="#000000",
                        alpha=0.8, label="Col={}.".format(col_idx))
            ax3.plot(p_col, x, c="#bc5090", lw=3,
                     label="Poly fit, order={}.".format(self.poly_order))

        ax3.set_ylabel("Pixel")
        ax3.set_xlabel("Electrons")
        ax3.legend(loc="upper center")

        plt.tight_layout()
        plt.show()

    def _draw_spatial_profile(self, int_idx):
        fig = plt.figure(figsize=(7, 7))
        ax1 = fig.add_subplot(111, projection="3d")
        row_pixel_vals = np.arange(0, self.P.shape[1])
        col_pixel_vals = np.arange(0, self.P.shape[2])
        xx, yy = np.meshgrid(col_pixel_vals, row_pixel_vals)
        ax1.plot_surface(xx, yy, self.P[int_idx], cmap="cividis", lw=0., rstride=1, cstride=1, alpha=0.9)
        ax1.set_xlabel("Pixel column")
        ax1.set_ylabel("Pixel row")
        ax1.set_zlabel("DN")
        plt.show()

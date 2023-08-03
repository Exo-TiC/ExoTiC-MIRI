import numpy as np
from jwst import datamodels
from jwst.stpipe import Step
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Extract1DOptimalStep(Step):
    """ Optimal extraction step. """

    spec = """
    median_spatial_profile = boolean(default=False)  # use median spatial profile.
    trace_position = string(default="constant")  # locate trace method: constant, gaussian_fits
    aperture_center = integer(default=36)  # center of aperture.
    aperture_left_width = integer(default=8)  # left-side of aperture width.
    aperture_right_width = integer(default=8)  # right-side of aperture width.
    draw_psf_fits = boolean(default=False)  # draw gauss fits to each column.
    draw_aperture = boolean(default=False)  # draw trace fits and position.
    draw_spectra = boolean(default=False)  # draw extracted spectra.
    """

    def process(self, input, wavelength_map, P, readnoise):
        """ Extract time-series 1D stellar spectra using optimal
        extraction as detailed in Horne 1986. The spatial profile
        must be pre-computed and input.

        Parameters
        ----------
        input: jwst.datamodels.CubeModel
            This is a rateints.fits loaded data segment.
        wavelength_map: np.ndarray
            The wavelength map. This is output from
            exotic_miri.reference.GetWavelengthMap.
        P: np.ndarray
            A cube of spatial profiles, one for each integration, of shape
            (n_ints, n_rows, n_cols). The should be normalised within
            each row (cross-disperion direction). These data can be made
            using exotic_miri.stage_2.CleanOutliersStep.
        readnoise: np.ndarray
            A cube of readnoise values, one for each integration, of shape
            (n_ints, n_rows, n_cols).
        median_spatial_profile: boolean
            If True median the spatial profiles through time. Default is
            False.
        trace_position: string
            The method for locating the spectral trace per detector row.
            constant: uses the value specified by aperture_center.
            gaussian_fits: fit a Gaussian to each row to find the centre.
        aperture_center: integer
            The defined centre of the spectral trace in terms of column
            index. Default is 36.
        aperture_left_width: integer
            The half-width of the box aperture in pixels away from the
            aperture_center. Default is 4, and so this aperture would
            include the aperture_center, say column 36, and 4 columns
            to the left of this.
        aperture_right_width: integer
            The half-width of the box aperture in pixels away from the
            aperture_center. Default is 4, and so this aperture would
            include the aperture_center, say column 36, and 4 columns
            to the right of this.
        draw_psf_fits: boolean
            Plot Gaussina fits to the PSF.
        draw_aperture: boolean
            Plot the defined aperture, within which optimal extraction
            is applied.
        draw_spectra: boolean
            Plot the extracted 1D spectra.

        Returns
        -------
        wavelengths, spectra, spectra_uncertainties, trace_widths: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            Arrays of wavelengths (n_rows,), spectra (n_ints, n_rows),
            spectra_uncertainties (n_ints, n_rows), trace_widths (n_ints,).

        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            extract_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error("Input is a {} which was not expected for "
                               "Extract1DBoxStep, skipping step.".format(
                                str(type(input_model))))
                return None, None, None

            # Define mask and spectral trace region.
            trace_mask_cube, trace_position, trace_sigmas = \
                self._define_spectral_trace_region(extract_model.data)
            extract_model.data = extract_model.data[trace_mask_cube].reshape(
                extract_model.data.shape[0], extract_model.data.shape[1], -1)
            extract_model.err = extract_model.err[trace_mask_cube].reshape(
                extract_model.data.shape[0], extract_model.data.shape[1], -1)
            P = P[trace_mask_cube].reshape(
                extract_model.data.shape[0], extract_model.data.shape[1], -1)
            readnoise = readnoise[trace_mask_cube].reshape(
                extract_model.data.shape[0], extract_model.data.shape[1], -1)
            if self.draw_aperture:
                self._draw_trace_mask(extract_model.data, trace_mask_cube)

            # Get wavelengths on trace.
            self.log.info("Assigning wavelengths using trace center.")
            wavelengths = wavelength_map[:, int(np.nanmedian(trace_position))]

            # Extract.
            self.log.info("Optimal extraction in progress.")
            if self.median_spatial_profile:
                P = np.median(P, axis=0)
                P /= np.sum(P, axis=1)[:, np.newaxis]
                P = np.broadcast_to(
                    P[np.newaxis, :, :], shape=extract_model.data.shape)
                self.log.info("Using median spatial profile.")
            else:
                # Re-norm rows in case different aperture width.
                P /= np.sum(P, axis=2)[:, :, np.newaxis]

            # Iterate integrations.
            fs_opt = []
            var_fs_opt = []
            for int_idx in range(extract_model.data.shape[0]):
                integration = extract_model.data[int_idx, :, :]
                variance = extract_model.err[int_idx, :, :]**2
                spatial_profile = P[int_idx, :, :]
                rn = readnoise[int_idx, :, :]

                # Extract standard spectrum.
                f, var_f = self._extract_standard_spectra(
                    integration, variance)

                # Revise variance estimate.
                var_revised = self._revise_variance_estimates(
                    f, spatial_profile, rn)

                # Extract optimal spectrum.
                f_opt, var_f_opt = self._extract_optimal_spectrum(
                    integration, spatial_profile, var_revised)
                fs_opt.append(f_opt)
                var_fs_opt.append(var_f_opt)

        fs_opt = np.array(fs_opt)
        var_fs_opt = np.array(var_fs_opt)

        if self.draw_spectra:
            self._draw_extracted_spectra(wavelengths, fs_opt)

        return wavelengths, fs_opt, var_fs_opt**0.5, trace_sigmas

    def _extract_standard_spectra(self, D_S, V):
        """ f and var_f as per Horne 1986 table 1 (step 4). """
        f = np.sum(D_S, axis=1)
        var_f = np.sum(V, axis=1)
        return f, var_f

    def _revise_variance_estimates(self, f, P, V_0, S=0., Q=1.):
        """ V revised as per Horne 1986 table 1 (step 6). """
        V_rev = V_0 + np.abs(f[:, np.newaxis] * P + S) / Q
        return V_rev

    def _extract_optimal_spectrum(self, D_S, P, V_rev):
        """ f optimal as per Horne 1986 table 1 (step 8). """
        f_opt = np.sum(P * D_S / V_rev, axis=1) / np.sum(P ** 2 / V_rev, axis=1)
        var_f_opt = np.sum(P, axis=1) / np.sum(P ** 2 / V_rev, axis=1)
        return f_opt, var_f_opt

    def _define_spectral_trace_region(self, data_cube):
        if self.trace_position == "constant":
            trace_position = np.zeros(data_cube.shape[0]) + self.aperture_center
            _, trace_sigmas = \
                self._find_trace_position_per_integration(data_cube)
        elif self.trace_position == "gaussian_fits":
            # Find trace position per integration with gaussian fits.
            trace_position, trace_sigmas = \
                self._find_trace_position_per_integration(data_cube)
        else:
            raise ValueError("locate_trace_method not recognised.")

        # Define trace region to be masked.
        trace_mask_cube = np.zeros(data_cube.shape).astype(bool)
        ints_mask_left_edge = np.rint(
            trace_position - self.aperture_left_width).astype(int)
        ints_mask_right_edge = np.rint(
            trace_position + self.aperture_right_width + 1).astype(int)
        for int_idx in range(trace_mask_cube.shape[0]):
            trace_mask_cube[int_idx, :, ints_mask_left_edge[
                int_idx]:ints_mask_right_edge[int_idx]] = True
        self.log.info("Trace mask made.")

        return trace_mask_cube, trace_position, trace_sigmas

    def _find_trace_position_per_integration(self, data_cube, sigma_guess=1.59):
        trace_position = []
        trace_sigmas = []
        col_pixels = np.arange(0, data_cube.shape[2], 1)
        for int_idx, int_data in enumerate(data_cube):

            # Median stack rows. TODO: make wv dep.
            median_row_data = np.median(int_data[200:390, 12:68], axis=0)
            col_pixels = np.arange(12, 68, 1)

            try:
                popt, pcov = curve_fit(
                    self._amp_gaussian, col_pixels, median_row_data,
                    p0=[np.max(median_row_data), col_pixels[np.argmax(median_row_data)],
                        sigma_guess, 0.], method="lm")
                trace_position.append(popt[1])
                trace_sigmas.append(popt[2])
                if self.draw_psf_fits:
                    self._draw_gaussian_fit(col_pixels, median_row_data, popt, pcov)
            except ValueError as err:
                self.log.warn("Gaussian fitting failed, nans present "
                              "for integration={}.".format(int_idx))
                trace_position.append(np.nan)
                trace_sigmas.append(np.nan)
            except RuntimeError as err:
                self.log.warn("Gaussian fitting failed to find optimal trace "
                              "centre for integration={}.".format(int_idx))
                trace_position.append(np.nan)
                trace_sigmas.append(np.nan)

        return np.array(trace_position), np.array(trace_sigmas)

    def _amp_gaussian(self, x_vals, a, mu, sigma, base=0.):
        y = a * np.exp(-(x_vals - mu)**2 / (2. * sigma**2))
        return base + y

    def _draw_gaussian_fit(self, x_data, y_data, popt, pcov):
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 7))

        # Data and fit.
        ax1.scatter(x_data, y_data, s=10, c="#000000",
                    label="Data")
        xs_hr = np.linspace(np.min(x_data), np.max(x_data), 1000)
        ax1.plot(xs_hr, self._amp_gaussian(
            xs_hr, popt[0], popt[1], popt[2], popt[3]), c="#bc5090",
                 label="Gaussian fit, mean={}.".format(popt[1]))

        # Gaussian centre and sigma.
        centre = popt[1]
        centre_err = np.sqrt(np.diag(pcov))[1]
        ax1.axvline(centre, ls="--", c="#000000")
        ax1.axvspan(centre - centre_err, centre + centre_err,
                    alpha=0.25, color="#000000")

        ax1.set_xlabel("Col pixels")
        ax1.set_ylabel("DN")
        ax1.set_title("$\mu$={}, and $\sigma$={}.".format(
            round(popt[1], 3), round(popt[2], 3)))
        plt.tight_layout()
        plt.show()

    def _draw_trace_mask(self, data_cube, trace_mask_cube):
        for int_idx in range(data_cube.shape[0]):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 7))
            ax1.get_shared_y_axes().join(ax1, ax2, ax3)
            ax1.get_shared_x_axes().join(ax1, ax2)

            # Data.
            im = data_cube[int_idx, :, :]
            ax1.imshow(im, origin="lower", aspect="auto", interpolation="none",
                       vmin=np.percentile(im.ravel(), 1.),
                       vmax=np.percentile(im.ravel(), 99.))

            # Mask.
            im = trace_mask_cube[int_idx, :, :]
            ax2.imshow(im, origin="lower", aspect="auto", interpolation="none")

            # Number of pixels.
            ax3.plot(np.sum(trace_mask_cube[int_idx, :, :], axis=1),
                     np.arange(trace_mask_cube.shape[1]))
            ax3.set_xlim(0, 72)

            fig.suptitle("Integration={}/{}.".format(
                int_idx, data_cube.shape[0]))
            ax1.set_ylabel("Row pixels")
            ax2.set_ylabel("Row pixels")
            ax3.set_ylabel("Row pixels")
            ax1.set_xlabel("Col pixels")
            ax2.set_xlabel("Col pixels")
            ax3.set_xlabel("Number of pixels")
            plt.tight_layout()
            plt.show()

    def _draw_extracted_spectra(self, wavelengths, spec_box):
        fig, ax1 = plt.subplots(1, 1, figsize=(13, 5))
        for int_idx in range(spec_box.shape[0]):
            ax1.plot(wavelengths, spec_box[int_idx, :], c="#bc5090", alpha=0.02)
        ax1.set_ylabel("Electrons")
        ax1.set_xlabel("Wavelength")
        plt.tight_layout()
        plt.show()

import numpy as np
from scipy import stats
from jwst import datamodels
from jwst.stpipe import Step
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Extract1dStep(Step):
    """ Extract 1d spectra.

    A step that will sky subtract, trace find, and extract 1d spectra
    from the 2d rate images using various algorithms. This step assumes
    the photom step has not been run, or at least input data units are
    DN/s.

    """

    spec = """
    bkg_algo = option("constant", "polynomial", default="polynomial")  # background algorithm
    bkg_region = int_list(default=None)  # background region, start, stop, start, stop
    bkg_poly_order = integer(default=1)  # order of polynomial for background fitting
    bkg_smoothing_length = integer(default=None)  # median smooth values over pixel length
    extract_algo = option("box", "optimal", "go", default="box")  # extraction algorithm
    extract_region_width = integer(default=20)  # full width of extraction region
    extract_poly_order = integer(default=1)  # order of polynomial for optimal extraction
    """

    reference_file_types = ['readnoise', 'gain']

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type CubeModel.

        Returns
        -------
        JWST data model
            A MultiSpecModel containing extracted 1d spectra, unless the
            step is skipped in which case `input_model` is returned.
        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            sci_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error('Input is a {} which was not expected for '
                               'extract_1d_step, skipping step.'.format(
                                str(type(input_model))))
                sci_model.meta.cal_step.extract_1d = 'SKIPPED'
                return sci_model

            # Check the observation mode.
            if not input_model.meta.exposure.type == 'MIR_LRS-SLITLESS':
                self.log.error('Observation is a {} which is not supported '
                               'by ExoTic-MIRIs extract_1d_step, skipping s'
                               'tep.'.format(input_model.meta.exposure.type))
                sci_model.meta.cal_step.extract_1d = 'SKIPPED'
                return sci_model

            # Get read noise data from reference file.
            read_noise_model_path = self.get_reference_file(
                input_model, 'readnoise')
            with datamodels.open(read_noise_model_path) as read_noise_model:
                read_noise_data = self._get_miri_subarray_data(
                    sci_model, read_noise_model)
                if read_noise_data is None:
                    sci_model.meta.cal_step.extract_1d = 'SKIPPED'
                    return sci_model

            # Get gain data from reference file.
            gain_model_path = self.get_reference_file(input_model, 'gain')
            with datamodels.open(gain_model_path) as gain_model:
                gain_data = self._get_miri_subarray_data(
                    sci_model, gain_model)
                if gain_data is None:
                    sci_model.meta.cal_step.extract_1d = 'SKIPPED'
                    return sci_model

            # Convert from DN/s (rate images) to DN.
            sci_model.data = sci_model.data \
                             * sci_model.meta.exposure.integration_time
            sci_model.err = sci_model.err \
                            * sci_model.meta.exposure.integration_time
            sci_model.meta.bunit_data = 'DN'

            # Background/sky subtract science data.
            bkg = self.compute_bkg_subtracted_data(
                sci_model.data, sci_model.err)
            if bkg is None:
                sci_model.meta.cal_step.extract_1d = 'SKIPPED'
                return sci_model

            # Extract 1d spectra.
            spectra, variances = self.extract_1d_spectra(
                D=sci_model.data,
                S=bkg,
                V=sci_model.err**2,
                V_0=read_noise_data**2,
                Q=gain_data)

            return spectra, variances
            for spec in spectra[0:10]:
                plt.plot(np.arange(spec.shape[0]), spec)
            plt.show()

            # TODO: build output data type, copy meta data,
            #  set spectra flux, pixels, and wavelengths.
            output_model = datamodels.MultiSpecModel()

            # Update meta.
            # Todo: update units.
            output_model.meta.cal_step.extract_1d = 'COMPLETE'
            output_model.meta.filetype = '1d spectrum'

        return output_model

    def _get_miri_subarray_data(self, sci_model, ref_model):
        """ Cutout data corresponding to MIRI subarray. """
        if sci_model.data.shape[1:] == ref_model.data.shape:
            return ref_model.data
        elif ref_model.data.shape == (1024, 1032):
            return ref_model.data[528:944, 0:72]
        else:
            self.log.error('Reference data model {} does not appear '
                           'compatible with the MIRI subarray.'.format(
                            ref_model))

    def compute_bkg_subtracted_data(self, sci_data, sci_err, draw=False):
        """ Compute/estimate background/sky data. """
        # Cutout data in specified background region.
        bkg = np.zeros(sci_data.shape)
        all_cols = np.arange(0, sci_data.shape[2])
        bkg_cols = np.r_[self.bkg_region[0]:self.bkg_region[1],
                         self.bkg_region[2]:self.bkg_region[3]]
        bkg_data = sci_data[:, :, bkg_cols]

        # Iterate integrations.
        self.log.info('Estimating and subtracting background for {} '
                      'integrations using the `{}` algorithm.'.format(
                       sci_data.shape[0], self.bkg_algo))
        for idx_int, integration in enumerate(bkg_data):

            if self.bkg_algo == 'constant':
                bkg_int = np.mean(stats.sigmaclip(
                    integration, low=5.0, high=5.0)[0])
                bkg[idx_int, :, :] = bkg_int

            elif self.bkg_algo == 'polynomial':
                bkg_int = []
                for idx_row, int_row in enumerate(integration):
                    p_coeff = np.polyfit(
                        bkg_cols, int_row, self.bkg_poly_order,
                        w=1/sci_err[idx_int, idx_row, bkg_cols])
                    bkg_int.append(np.polyval(p_coeff, all_cols))

                    if draw:
                        self._draw_bkg_poly_fits(
                            bkg_cols, int_row, all_cols,
                            np.polyval(p_coeff, all_cols),
                            idx_row)

                bkg_int = np.array(bkg_int)
                if self.bkg_smoothing_length is not None:
                    bkg_int = self._median_smooth_per_column(bkg_int)
                bkg[idx_int, :, :] = bkg_int

            else:
                self.log.error('Background algorithm not supported. See '
                               'bkg_algo = option("constant", "polynomia'
                               'l", default="polynomial")')
                return None

        return bkg

    def _median_smooth_per_column(self, bkg):
        """ Median smooth data per column. """
        sm_bkg = np.copy(bkg)
        n_rows = bkg.shape[0]
        sm_radius = int((self.bkg_smoothing_length - 1) / 2)
        for idx_row in range(n_rows):
            # Define window.
            start_row = max(0, idx_row - sm_radius)
            end_row = min(n_rows - 1, idx_row + sm_radius)

            # Compute median in window.
            sm_bkg[idx_row, :] = np.median(
                bkg[start_row:end_row + 1, :], axis=0)

        return sm_bkg

    def _draw_bkg_poly_fits(self, x_data, y_data, x_model, y_model, idx_slice):
        """ Draw the polynomial fits to the background. """
        fig = plt.figure(figsize=(8, 7))
        ax1 = fig.add_subplot(111)
        ax1.scatter(x_data, y_data, s=10, c='#000000',
                    label='Bkg pixels, slice={}.'.format(idx_slice))
        ax1.plot(x_model, y_model, c='#bc5090',
                 label='Poly fit, order={}.'.format(self.bkg_poly_order))
        ax1.set_xlabel('Pixel row')
        ax1.set_ylabel('DN/s')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def extract_1d_spectra(self, D, S, V, V_0, Q):
        """ Extract 1d spectra from bkg subtracted science data. """
        self.log.info('Extracting 1d spectra for {} integrations using the '
                      '`{}` algorithm.'.format(D.shape[0], self.extract_algo))
        if self.extract_algo == 'box':
            return self._box_extraction(D, S, V)

        elif self.extract_algo == 'optimal':
            return self._optimal_extraction(D, S, V, V_0, Q)

        elif self.extract_algo == 'go':
            # NB. experimental Global Optimal extraction
            # Potentially we want to cut a border away from the edge too.
            # Then on shift-reg-grid we cut to the data strip we care about.
            raise NotImplementedError('Experimental algo dev WIP.')

        else:
            self.log.error('Extract 1d algorithm not supported. See '
                           'extract_algo = option("box", "optimal", '
                           '"go", default="box").')
            return None, None

    def _box_extraction(self, D, S, V):
        """ Extract with fixed-width top-hat aperture. """
        # Subtract background/sky.
        D_S = D - S

        # Iterate integrations.
        all_cols = np.arange(0, D.shape[2])
        fs = []
        var_fs = []
        for integration, variance in zip(D_S, V):

            # Find spectral trace location in x (column position). Assumes
            # not rotation, ie. one x position for entire trace. Stack rows
            # for master psf of integration and fit Gaussian for the centre.
            psf_int = np.sum(integration, axis=0)
            psf_centre = self._get_fitted_gaussian_centre(
                all_cols, psf_int, sigma_guess=3., fit_region_width=11)

            # Define extraction region by column idxs.
            region_start_idx, region_end_idx = \
                self._get_start_and_end_idxs_of_region(psf_centre)

            # Extract standard spectrum.
            f, var_f = self._extract_standard_spectra(
                integration, variance, region_start_idx, region_end_idx)
            fs.append(f)
            var_fs.append(var_f)

        return np.array(fs), np.array(var_fs)

    def _optimal_extraction(self, D, S, V, V_0, Q):
        """ Extract using optimal extraction method of Horne 1986. """
        # Subtract background/sky.
        D_S = D - S

        # Iterate integrations.
        all_cols = np.arange(0, D.shape[2])
        fs_opt = []
        var_fs_opt = []
        for integration, bkg, variance in zip(D_S, S, V):

            # Find spectral trace location in x (column position). Assumes
            # not rotation, ie. one x position for entire trace. Stack rows
            # for master psf of integration and fit Gaussian for the centre.
            psf_int = np.sum(integration, axis=0)
            psf_centre = self._get_fitted_gaussian_centre(
                all_cols, psf_int, sigma_guess=3., fit_region_width=11)

            # Define extraction region by column idxs.
            region_start_idx, region_end_idx = \
                self._get_start_and_end_idxs_of_region(psf_centre)

            # Extract standard spectrum.
            f, var_f = self._extract_standard_spectra(
                integration, variance, region_start_idx, region_end_idx)

            # Construct spatial profile.
            P = self._construct_spatial_profile(
                integration, variance, region_start_idx, region_end_idx)
            P /= np.sum(P, axis=1)[:, np.newaxis]

            # Revise variance estimate.
            var_revised = self._revise_variance_estimates(
                f, bkg, P, V_0, Q, region_start_idx, region_end_idx)

            # Extract optimal spectrum.
            f_opt, var_f_opt = self._extract_optimal_spectrum(
                integration, P, var_revised, region_start_idx, region_end_idx)
            fs_opt.append(f_opt)
            var_fs_opt.append(var_f_opt)

        return np.array(fs_opt), np.array(var_fs_opt)

    def _get_fitted_gaussian_centre(self, xs, ys, sigma_guess=3.,
                                    fit_region_width=None, draw=False):
        """ Get centre by fitting a Gaussian function. """
        if fit_region_width is not None:
            # Trim region for fitting.
            idx_peak = np.argmax(ys)
            fit_region_radius = int((fit_region_width - 1) / 2)
            xs = xs[idx_peak - fit_region_radius:
                    idx_peak + fit_region_radius + 1]
            ys = ys[idx_peak - fit_region_radius:
                    idx_peak + fit_region_radius + 1]

        try:
            # Least squares fit data.
            popt, pcov = curve_fit(
                self._amp_gaussian, xs, ys,
                p0=[np.max(ys), xs[np.argmax(ys)], sigma_guess])
        except ValueError as err:
            self.log.error('Gaussian fitting to find centre failed.')
            return None

        if draw:
            self._draw_gaussian_centre_fits(xs, ys, popt)

        return popt[1]

    def _draw_gaussian_centre_fits(self, x_data, y_data, popt):
        """ Draw Gaussian fits for centre finding. """
        fig = plt.figure(figsize=(8, 7))
        ax1 = fig.add_subplot(111)
        ax1.scatter(x_data, y_data, s=10, c='#000000',
                    label='Data')
        xs_hr = np.linspace(np.min(x_data), np.max(x_data), 1000)
        ax1.plot(xs_hr, self._amp_gaussian(
            xs_hr, popt[0], popt[1], popt[2]), c='#bc5090',
                 label='Gaussian fit, mean={}.'.format(popt[1]))
        ax1.axvline(popt[1], ls='--')
        ax1.set_xlabel('Pixels')
        ax1.set_ylabel('DN')
        plt.tight_layout()
        plt.show()

    def _amp_gaussian(self, x_vals, a, mu, sigma):
        """ Scalable Gaussian function. """
        y = a * np.exp(-(x_vals - mu)**2 / (2. * sigma**2))
        return y

    def _get_start_and_end_idxs_of_region(self, psf_centre):
        """ Get start and end idxs of region. """
        region_radius = int((self.extract_region_width - 1) / 2)
        region_start = int(round(psf_centre - region_radius))
        region_end = int(round(psf_centre + region_radius + 1))
        return region_start, region_end

    def _extract_standard_spectra(self, D_S, V, region_start_idx,
                                  region_end_idx):
        """ f and var_f as per Horne 1986 table 1 (step 4). """
        if D_S.ndim == 2:
            f = np.sum(D_S[:, region_start_idx:region_end_idx], axis=1)
            var_f = np.sum(V[:, region_start_idx:region_end_idx], axis=1)
            return f, var_f

        elif D_S.ndim == 3:
            f = np.sum(D_S[:, :, region_start_idx:region_end_idx],
                       axis=2)
            var_f = np.sum(V[:, :, region_start_idx:region_end_idx],
                           axis=2)
            return f, var_f

        else:
            self.log.error('Psf centres input type not recognised.')
            return None, None

    def _construct_spatial_profile(self, D_S, V, region_start_idx,
                                   region_end_idx, draw=False):
        """ P as per Horne 1986 table 1 (step 5). """
        P = []

        # Iterate columns.
        row_pixel_idxs = np.arange(D_S.shape[0])
        for col_idx in np.arange(region_start_idx, region_end_idx):

            D_S_col = np.copy(D_S[:, col_idx])
            D_S_col_mask = np.ones(D_S_col.shape[0])
            V_col = np.copy(V[:, col_idx])
            while True:
                # Replace flagged pixels with nearby median.
                frr = 10
                for flag_idx in np.where(D_S_col_mask == 0)[0]:
                    D_S_col[flag_idx] = np.median(
                        D_S_col[max(0, flag_idx - frr):
                              min(D_S_col.shape[0] - 1, flag_idx + frr + 1)])

                # Fit polynomial to column.
                p_coeff = np.polyfit(row_pixel_idxs, D_S_col,
                                     self.extract_poly_order, w=1/V_col**0.5)
                p_col = np.polyval(p_coeff, row_pixel_idxs)

                # Check residuals to polynomial fit.
                res_col = D_S_col - p_col
                dev_col = np.abs(res_col) / np.std(res_col)
                max_deviation_idx = np.argmax(dev_col)
                if dev_col[max_deviation_idx] > 7.:
                    # Outlier: mask and repeat poly fitting.
                    D_S_col_mask[max_deviation_idx] = 0
                    continue
                else:
                    P.append(p_col)
                    if draw:
                        self._draw_bkg_poly_fits(
                            row_pixel_idxs, D_S_col,
                            row_pixel_idxs, p_col, col_idx)
                    break

        # Rotate.
        P = np.array(P).T

        # Enforce positivity.
        P[P < 0] = 0.

        return P

    def _revise_variance_estimates(self, f, S, P, V_0, Q, region_start_idx,
                                   region_end_idx):
        """ V revised as per Horne 1986 table 1 (step 6). """
        V_rev = V_0[:, region_start_idx:region_end_idx] + np.abs(
            f[:, np.newaxis] * P + S[:, region_start_idx:region_end_idx]) \
                / Q[:, region_start_idx:region_end_idx]
        return V_rev

    def _extract_optimal_spectrum(self, D_S, P, V_rev, region_start_idx,
                                  region_end_idx):
        """ f optimal as per Horne 1986 table 1 (step 8). """
        f_opt = np.sum(P * D_S[:, region_start_idx:region_end_idx] / V_rev,
                       axis=1) / np.sum(P**2 / V_rev, axis=1)
        var_f_opt = np.sum(P, axis=1) / np.sum(P**2 / V_rev, axis=1)
        return f_opt, var_f_opt

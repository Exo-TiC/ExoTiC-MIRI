import warnings
import numpy as np
import pandas as pd
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
    extract_algo = option("box", "optimal", "anchor", default="box")  # extraction algorithm
    extract_region_width = integer(default=20)  # full width of extraction region
    extract_poly_order = integer(default=1)  # order of polynomial for optimal extraction
    max_iter = integer(default=10)  # max iterations of anchor algorithm
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
            A MultiSpecModel containing extracted 1d spectra. The
            spectra for each integration are packaged as a list of
            pandas.DataFrames in MultiSpecModel.spectra.
            If the step is skipped the `input_model` is returned.
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
            with datamodels.ReadnoiseModel(read_noise_model_path) \
                    as read_noise_model:
                read_noise_data = self._get_miri_subarray_data(
                    sci_model, read_noise_model)
                # Todo: care w/ flagged pixels in sci and ref data.
                read_noise_data[read_noise_data == 1000] = \
                    np.median(read_noise_data)
                if read_noise_data is None:
                    sci_model.meta.cal_step.extract_1d = 'SKIPPED'
                    return sci_model

            # Get gain data from reference file.
            gain_model_path = self.get_reference_file(input_model, 'gain')
            with datamodels.GainModel(gain_model_path) as gain_model:
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

            # Extract 1d spectra, variances, and spectral trace shifts.
            spectra, variances = self.extract_1d_spectra(
                D=sci_model.data,
                S=bkg,
                V=sci_model.err**2,
                V_0=read_noise_data**2,
                Q=gain_data)

            # Link with WCS: transform pixels to wavelengths.
            pixels, wavelengths = self._link_world_coordinate_system(
                input_model)

            # Package results.
            output_model = self._package_compatible_multispec_datamodel(
                pixels, wavelengths, spectra, variances, input_model)

            # Update meta.
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

        elif self.extract_algo == 'anchor':
            # NB. experimental anchor extraction.
            return self._anchor_extraction(D, S, V, V_0, Q)

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
            # no rotation, ie. one x position for entire trace. Stack rows
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
            # no rotation, ie. one x position for entire trace. Stack rows
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
            if np.isnan(P).any():
                self.log.error(
                    'Spatial profile contains entire slice of negative '
                    'values. Background estimation needs to be improved'
                    '. Try the bkg_algo=`polynomial` algorithm.')
                return None, None

            # Revise variance estimate.
            var_revised = self._revise_variance_estimates(
                f, bkg, P, V_0, Q, region_start_idx, region_end_idx)

            # Extract optimal spectrum.
            f_opt, var_f_opt = self._extract_optimal_spectrum(
                integration, P, var_revised, region_start_idx, region_end_idx)
            fs_opt.append(f_opt)
            var_fs_opt.append(var_f_opt)

        return np.array(fs_opt), np.array(var_fs_opt)

    def _anchor_extraction(self, D, S, V, V_0, Q, anchor_int=0):
        """ Extract using anchor extraction method. NB. experimental. """
        # Subtract background/sky.
        D_S = D - S

        # Setup data structures.
        all_cols = np.arange(0, D.shape[2])
        all_rows = np.arange(0, D.shape[1])
        x_shifts = np.zeros(D.shape[0])
        y_shifts = np.zeros(D.shape[0])
        original_grids = {'D_S': D_S, 'S': S, 'V': V,
                          'V_0': np.tile(V_0, (D.shape[0], 1, 1)),
                          'Q': np.tile(Q, (D.shape[0], 1, 1))}

        # Define anchor position. Given by anchor_int' trace.
        # The spectral extraction region and trace location. All
        # integrations' traces will be shift-re-gridded onto here.
        psf_int_anchor = np.sum(D_S[anchor_int], axis=0)
        psf_centre_anchor = self._get_fitted_gaussian_centre(
            all_cols, psf_int_anchor, sigma_guess=3., fit_region_width=11)
        region_start_idx, region_end_idx = \
            self._get_start_and_end_idxs_of_region(psf_centre_anchor)
        region_pix_width = region_end_idx - region_start_idx
        re_grids_xr = {'D_S': np.empty(D_S.shape[0:2] + (region_pix_width,)),
                       'S': np.empty(S.shape[0:2] + (region_pix_width,)),
                       'V': np.empty(V.shape[0:2] + (region_pix_width,)),
                       'V_0': np.empty(V.shape[0:2] + (region_pix_width,)),
                       'Q': np.empty(V.shape[0:2] + (region_pix_width,))}

        # Iterate until convergence. Shift-re-gridding towards a global
        # spatial profile, anchored in place.
        iter = 0
        while True:

            # Re-grid based on latest shift updates. Initial re-grid
            # leaves the arrays identical to the original.
            for idx in range(D.shape[0]):

                # Update row and col pixel values with shift for this
                # integration. Only for anchored extraction region.
                shifted_rows_xr = all_rows + y_shifts[idx]
                shifted_cols_xr = all_cols[region_start_idx: region_end_idx] \
                                  + x_shifts[idx]

                # Interpolate each data array onto anchored extraction region.
                for key, og_grid in original_grids.items():
                    interp_grid = interpolate.RectBivariateSpline(
                        all_rows, all_cols, og_grid[idx, :, :], kx=3, ky=3)
                    re_gridded = interp_grid(shifted_rows_xr, shifted_cols_xr)
                    if key in ['V', 'V_0', 'Q']:
                        # Enforce variance and gain positivity.
                        re_gridded[re_gridded < 0.] = 0.
                    re_grids_xr[key][idx, :, :] = re_gridded

            # Stack all integrations.
            stacked_D_S = np.sum(re_grids_xr['D_S'], axis=0)
            stacked_V = np.sum(re_grids_xr['V'], axis=0)

            # Construct global spatial profiles for each direction.
            P_global = self._construct_spatial_profile(
                stacked_D_S, stacked_V, 0, region_pix_width)
            P_global_spec = P_global / np.sum(P_global, axis=1)[:, np.newaxis]
            P_global_psf = P_global / np.sum(P_global, axis=0)[np.newaxis, :]
            if np.isnan(P_global_spec).any() or np.isnan(P_global_psf).any():
                self.log.error(
                    'Spatial profile contains entire slice of negative '
                    'values. Background estimation needs to be improved'
                    '. Try the bkg_algo=`polynomial` algorithm.')
                return None, None

            # Optimal extract both spectra (fs, dispersion direction) and
            # psfs (gs, cross-dispersion direction) using global spatial
            # profile in anchored region.
            fs, var_fs = self._extract_standard_spectra(
                re_grids_xr['D_S'], re_grids_xr['V'])
            var_fs_revised = self._revise_variance_estimates(
                fs, re_grids_xr['S'], P_global_spec, re_grids_xr['V_0'],
                re_grids_xr['Q'], mode='spec')
            fs_opt, var_fs_opt = self._extract_optimal_spectrum(
                re_grids_xr['D_S'], P_global_spec, var_fs_revised, mode='spec')

            gs, var_gs = self._extract_standard_psf(
                re_grids_xr['D_S'], re_grids_xr['V'])
            var_gs_revised = self._revise_variance_estimates(
                gs, re_grids_xr['S'], P_global_psf, re_grids_xr['V_0'],
                re_grids_xr['Q'], mode='psf')
            gs_opt, var_gs_opt = self._extract_optimal_spectrum(
                re_grids_xr['D_S'], P_global_psf, var_gs_revised, mode='psf')

            # Compute spectral trace shifts.
            template_spec = np.median(fs_opt, axis=0)
            positions_spec = self._compute_cross_correlation_positions(
                all_rows, template_spec, fs_opt,
                sigma_guess=9., fit_region_width=33)
            shifts_spec_corr = positions_spec - positions_spec[anchor_int]
            y_shifts += shifts_spec_corr

            template_psf = np.median(gs_opt, axis=0)
            positions_psf = self._compute_cross_correlation_positions(
                all_cols[region_start_idx: region_end_idx],
                template_psf, gs_opt,
                sigma_guess=3., fit_region_width=11)
            shifts_psf_corr = positions_psf - positions_psf[anchor_int]
            x_shifts += shifts_psf_corr

            # Test for convergence.
            iter += 1
            self.log.info('Anchor shift reporting: x shifts median={}, max={}'
                          ' and y shifts median={} max={}.'.format(
                           round(np.median(np.abs(shifts_psf_corr)), 6),
                           round(np.max(np.abs(shifts_psf_corr)), 6),
                           round(np.median(np.abs(shifts_spec_corr)), 6),
                           round(np.max(np.abs(shifts_spec_corr)), 6)))
            if np.all(np.abs(shifts_spec_corr) < 1e-3) \
                    and np.all(np.abs(shifts_psf_corr) < 1e-3):
                self.log.info('Anchor convergence reached in {} iterations.'
                              .format(iter))
                return fs_opt, var_fs_opt
            if iter >= self.max_iter:
                self.log.info('Anchor max {} iterations reached.'
                              .format(iter))
                return fs_opt, var_fs_opt

    def _get_fitted_gaussian_centre(self, xs, ys, sigma_guess=3.,
                                    fit_region_width=11., draw=False):
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
        ax1.set_title('Amp={}, $\mu$={}, and $\sigma$={}.'.format(
            round(popt[0], 3), round(popt[1], 3), round(popt[2], 3)))
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

    def _compute_cross_correlation_positions(self, pixels, template, dataset,
                                             sigma_guess=3.,
                                             fit_region_width=11, draw=False):
        """ Compute positions from cross-correlation functions. """
        # Standardise.
        template_norm = template / np.max(template)
        dataset_norm = dataset / np.max(dataset, axis=1)[:, np.newaxis]

        shifts = []
        for spec in dataset_norm:

            # Cross-correlation with template.
            ccf = np.correlate(spec, template_norm, mode='same')

            # Find shifts from Gaussian fits to the centre of each
            # cross-correlation function.
            shifts.append(self._get_fitted_gaussian_centre(
                pixels, ccf,
                sigma_guess=sigma_guess,
                fit_region_width=fit_region_width,
                draw=draw))

        return np.array(shifts)

    def _extract_standard_spectra(self, D_S, V, region_start_idx=None,
                                  region_end_idx=None):
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
            self.log.error('Extract standard spectra input shape '
                           'not recognised.')
            return None, None

    def _extract_standard_psf(self, D_S, V, region_start_idx=None,
                              region_end_idx=None):
        """ Similar to standard spectra but cross-dispersion profiles. """
        if D_S.ndim == 2:
            g = np.sum(D_S[:, region_start_idx:region_end_idx], axis=0)
            var_g = np.sum(V[:, region_start_idx:region_end_idx], axis=0)
            return g, var_g

        elif D_S.ndim == 3:
            g = np.sum(D_S[:, :, region_start_idx:region_end_idx],
                       axis=1)
            var_g = np.sum(V[:, :, region_start_idx:region_end_idx],
                           axis=1)
            return g, var_g

        else:
            self.log.error('Extract standard psf input shape '
                           'not recognised.')
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
                try:
                    p_coeff = np.polyfit(row_pixel_idxs, D_S_col,
                                         self.extract_poly_order, w=1/V_col**0.5)
                except np.linalg.LinAlgError as err:
                    self.log.error('Poly fit error when constructing '
                                   'spatial profile.')
                    return None
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
        P[P < 0.] = 0.

        return P

    def _revise_variance_estimates(self, f, S, P, V_0, Q, region_start_idx=None,
                                   region_end_idx=None, mode='spec'):
        """ V revised as per Horne 1986 table 1 (step 6). """
        if f.ndim == 1:
            if mode == 'spec':
                V_rev = V_0[:, region_start_idx:region_end_idx] + np.abs(
                    f[:, np.newaxis] * P
                    + S[:, region_start_idx:region_end_idx]) \
                    / Q[:, region_start_idx:region_end_idx]
                return V_rev

            elif mode == 'psf':
                V_rev = V_0[:, region_start_idx:region_end_idx] + np.abs(
                    f[np.newaxis, :] * P
                    + S[:, region_start_idx:region_end_idx]) \
                    / Q[:, region_start_idx:region_end_idx]
                return V_rev

            else:
                self.log.error('Revise variance mode not recognised.')
                return None

        elif f.ndim == 2:
            if mode == 'spec':
                V_rev = V_0[:, :, region_start_idx:region_end_idx] + np.abs(
                    f[:, :, np.newaxis] * P[np.newaxis, :, :]
                    + S[:, :, region_start_idx:region_end_idx]) \
                    / Q[:, :, region_start_idx:region_end_idx]
                return V_rev

            elif mode == 'psf':
                V_rev = V_0[:, :, region_start_idx:region_end_idx] + np.abs(
                    f[:, np.newaxis, :] * P[np.newaxis, :, :]
                    + S[:, :, region_start_idx:region_end_idx]) \
                    / Q[:, :, region_start_idx:region_end_idx]
                return V_rev

            else:
                self.log.error('Revise variance mode not recognised.')
                return None

        else:
            self.log.error('Revise variance estimates, data input '
                           'shape not recognised.')
            return None

    def _extract_optimal_spectrum(self, D_S, P, V_rev, region_start_idx=None,
                                  region_end_idx=None, mode='spec'):
        """ f optimal as per Horne 1986 table 1 (step 8). """
        if D_S.ndim == 2:
            if mode == 'spec':
                f_opt = np.sum(
                    P * D_S[:, region_start_idx:region_end_idx] / V_rev, axis=1) \
                    / np.sum(P**2 / V_rev, axis=1)
                var_f_opt = np.sum(P, axis=1) / np.sum(P**2 / V_rev, axis=1)
                return f_opt, var_f_opt

            elif mode == 'psf':
                f_opt = np.sum(
                    P * D_S[:, region_start_idx:region_end_idx] / V_rev, axis=0) \
                    / np.sum(P**2 / V_rev, axis=0)
                var_f_opt = np.sum(P, axis=0) / np.sum(P**2 / V_rev, axis=0)
                return f_opt, var_f_opt

            else:
                self.log.error('Extract optimal mode not recognised.')
                return None, None

        elif D_S.ndim == 3:
            if mode == 'spec':
                f_opt = np.sum(
                    P[np.newaxis, :, :]
                    * D_S[:, :, region_start_idx:region_end_idx] / V_rev, axis=2) \
                    / np.sum(P[np.newaxis, :, :]**2 / V_rev, axis=2)
                var_f_opt = np.sum(P[np.newaxis, :, :], axis=2) \
                            / np.sum(P[np.newaxis, :, :]**2 / V_rev, axis=2)
                return f_opt, var_f_opt

            elif mode == 'psf':
                f_opt = np.sum(
                    P[np.newaxis, :, :]
                    * D_S[:, :, region_start_idx:region_end_idx] / V_rev, axis=1) \
                    / np.sum(P[np.newaxis, :, :]**2 / V_rev, axis=1)
                var_f_opt = np.sum(P[np.newaxis, :, :], axis=1) \
                            / np.sum(P[np.newaxis, :, :]**2 / V_rev, axis=1)
                return f_opt, var_f_opt

            else:
                self.log.error('Extract optimal mode not recognised.')
                return None, None

        else:
            self.log.error('Extract optimal spectrum, data input '
                           'shape not recognised.')
            return None, None

    def _link_world_coordinate_system(self, input_model):
        """ Link WCS and find pixels to wavelengths. """
        self.log.info('Mapping pixels to wavelengths for spectra.')

        # Build wavelength map.
        row_g, col_g = np.mgrid[0:input_model.data.shape[1],
                                0:input_model.data.shape[2]]
        wavelength_map = input_model.meta.wcs(
            col_g.ravel(), row_g.ravel())[-1]\
            .reshape(input_model.data.shape[1:])

        # Compute wavelengths as mean within each row.
        pixels = np.arange(0, input_model.data.shape[1])
        with warnings.catch_warnings():
            # All nan slice gives warning. Prevent the warning.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            wavelengths = np.nanmean(wavelength_map, axis=1)

        return pixels, wavelengths

    def _package_compatible_multispec_datamodel(self, pixels, wavelengths,
                                                spectra, variances,
                                                input_model):
        """ Build a multispec data structure compatible w/ STScI pipeline. """
        self.log.info('Packaging results at datamodels.MultiSpecModel'
                      '.spectra as list of pandas.DataFrames.')

        # Instantiate MultiSpecModel.
        output_model = datamodels.MultiSpecModel()

        # Copy meta data across.
        if hasattr(input_model, 'int_times'):
            output_model.int_times = input_model.int_times.copy()
        output_model.update(input_model, only='PRIMARY')

        # Iterate integrations.
        output_model.spectra = []
        for spec_int, var_int in zip(spectra, variances):

            # Build dataframe.
            spec_df = pd.DataFrame()
            spec_df['pixels'] = pixels
            spec_df['wavelengths'] = wavelengths
            spec_df['flux'] = spec_int
            spec_df['flux_error'] = var_int**0.5
            output_model.spectra.append(spec_df)

        return output_model

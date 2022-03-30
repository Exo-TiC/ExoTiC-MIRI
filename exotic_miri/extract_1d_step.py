import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class Extract1dStep(Step):
    """ Extract 1d spectra.

    A step that will sky subtract, trace find, and extract 1d spectra
    from the 2d rate images using various algorithms.

    """

    spec = """
    bkg_algo = option("constant", "polynomial", default="polynomial")  # background algorithm
    bkg_region = int_list(default=None)  # background region, start, stop, start, stop
    bkg_poly_order = integer(default=1)  # order of polynomial for background fitting
    bkg_smooth = integer(default=None)  # median smooth values over pixel length
    extract_algo = option("box", "optimal", "go", default="box")  # extraction algorithm
    extract_region_width = integer(default=20)  # full width of extraction region
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
                read_noise_data = self.get_miri_subarray_data(
                    sci_model, read_noise_model)

            # Get gain data from reference file.
            gain_model_path = self.get_reference_file(input_model, 'gain')
            with datamodels.open(gain_model_path) as gain_model:
                gain_data = self.get_miri_subarray_data(
                    sci_model, gain_model)

            print(read_noise_data)
            print(read_noise_data.shape)
            print(gain_data)
            print(gain_data.shape)

            # TODO: do extraction. diff types
            # Todo -- DN vs e- vs rates for input data, func of 2d.
            # Todo -- Sky background subtraction in 2d.
            # Todo -- non-parametric vs polynomial fits to background and weightings model.
            # Todo -- convergence criterion: all or median < eps.
            # Todo -- make a notebook example of the three methods.
            # Todo -- how can we do it for all integrations, or just per data chunk?
            # Todo -- implement as a step.

            # TODO: build output data type.

            # Update meta.
            sci_model.meta.cal_step.extract_1d = 'COMPLETE'
            sci_model.meta.filetype = '1d spectrum'

        return sci_model

    def get_miri_subarray_data(self, sci_model, ref_model):
        """ Cutout data corresponding to MIRI subarray. """
        if sci_model.data.shape[1:] == ref_model.data.shape:
            return ref_model.data
        elif ref_model.data.shape == (1024, 1032):
            return ref_model.data[528:944, 0:72]
        else:
            self.log.error('Reference data model {} does not appear '
                           'compatible with the MIRI subarray.'.format(
                            ref_model))

    def load_rate_images(self):
        # Load as many rate images as possible.
        # Ideally we want a global solution for the entire
        # dataset, but may have to be per data chunk.
        # Also load the error arrays.
        # Perhaps can process per chunk but maintaining global knowledge in
        # some way. ie. summing overall counts but keeping shift arrays for all.

        # Potentially we want to cut a border away from the edge too.
        # Then on shift-reg-grid we cut to the data strip we care about.
        return

    def load_referece_files(self):
        # Load the read noise ref file.
        # Load the gain ref file.
        return

    def convert_to_data_numbers(self):
        # Get integration durations.
        # Convert data and err arrays from rates to data numbers.
        return

    def subtract_background(self):
        # Subtract background from all rate images.
        # Update err arrays. Wait no updates needed for subtraction if background well known.
        # Options: constant, polynomial per row, smoothed, gp.
        return

    def extract_spec(self):
        # Option 1.
        # Box.

        # Option 2.
        # Optimal extraction per frame.

        # Option 3.
        # Global optimal extraction.
        return

    def box_extraction(self):
        return







# import numpy as np
# from scipy import interpolate
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
#
# from jwst.pipeline import calwebb_detector1
# from jwst.pipeline import calwebb_spec2
#
# def gaussian(x_vals, mu, sigma):
#     y = 1 / (sigma * (2 * np.pi) ** 0.5) * np.exp(-(x_vals - mu) ** 2 / (2. * sigma ** 2))
#     return y
#
# def amp_gaussian(x_vals, a, mu, sigma):
#     y = a * np.exp(-(x_vals - mu) ** 2 / (2. * sigma ** 2))
#     return y
#
# # Reproducible.
# np.random.seed(111)
#
# # Config.
# n_rows = 416
# n_cols = 68
# n_ints = 100
# row_pixel_vals = np.arange(0, n_rows)
# col_pixel_vals = np.arange(0, n_cols)
# psf_locs = np.linspace(33.0, 35.0, n_ints) + np.random.normal(loc=0., scale=0.1, size=n_ints)
# psf_sigma_func = np.linspace(1.25, 1.35, n_rows)
# spec_func_amp = 2.e6
# spec_func_sigma = 75.
# spec_func_means = np.linspace(211., 211.0, n_ints) + np.random.normal(loc=0., scale=0.1, size=n_ints)
# Q = 1
# bias = 150.
# readout_noise = 20.
# n_bins = 5
# bins = np.linspace(0, n_rows - 1, n_bins + 1).astype(np.int64)
#
# def make_integration_frame(spec_loc, psf_loc, draw=False):
#     data_2d = np.zeros((n_rows, n_cols), dtype=np.double)
#     for row_idx in row_pixel_vals:
#         data_2d[row_idx] = gaussian(col_pixel_vals, psf_loc, psf_sigma_func[row_idx])
#     data_2d = np.array(data_2d)
#     spec_func = spec_func_amp * gaussian(row_pixel_vals, spec_loc, spec_func_sigma)
#     data_2d = data_2d * spec_func[:, np.newaxis]
#     data_2d = np.random.poisson(data_2d)
#     data_2d = data_2d + bias
#     data_2d += np.random.normal(loc=0., scale=readout_noise, size=data_2d.shape)
#     variance_2d = readout_noise ** 2 + np.abs(data_2d) / Q
#
#     if draw:
#         fig = plt.figure(figsize=(9, 8))
#         ax1 = fig.add_subplot(111, projection='3d')
#         xx, yy = np.meshgrid(col_pixel_vals, row_pixel_vals)
#         ax1.plot_surface(xx, yy, data_2d, cmap='cividis', lw=0., rstride=1, cstride=1, alpha=0.9)
#         # ax1.set_box_aspect((np.ptp(xx), np.ptp(yy), 100))
#         ax1.set_xlabel('Pixel column')
#         ax1.set_ylabel('Pixel row')
#         ax1.set_zlabel('DN')
#         plt.show()
#
#     return data_2d, variance_2d
#
# def build_global_weightings_img(d_strip, var_strip):
#     strip_width = 10
#     p_initial = np.copy(d_strip)
#     poly_fit_weights = 1 / var_strip ** 0.5
#
#     p = []
#     for col in col_pixel_vals[:strip_width * 2]:
#         p_iter = np.copy(p_initial)
#         p_mask = np.ones(row_pixel_vals.shape)
#         while True:
#             for idx in np.where(p_mask == 0)[0]:
#                 p_iter[idx] = np.median(p_iter[np.max((0, idx - 10)): idx + 11])
#             p_coeff = np.polyfit(row_pixel_vals, p_iter[:, col], 10, w=poly_fit_weights[:, col])
#             p_model = np.polyval(p_coeff, row_pixel_vals)
#             residuals = p_iter[:, col] - p_model
#             deviations = np.abs(residuals) / np.std(residuals)
#             max_deviation_idx = np.argmax(deviations)
#             if deviations[max_deviation_idx] > 7.:
#                 p_mask[max_deviation_idx] = 0
#                 continue
#             else:
#                 p.append(p_model)
#                 # plt.plot(p_iter[:, col])
#                 # plt.plot(p_model)
#                 # plt.show()
#                 break
#
#     p = np.array(p).T
#     p[p < 0] = 0.
#     p_spec = p / np.sum(p, axis=1)[:, np.newaxis]
#     p_psf = p / np.sum(p, axis=0)[np.newaxis, :]
#     return p_spec, p_psf
#
# def cross_correlation(pixels, template, dataset, width=5):
#     # Standardise.
#     template /= np.max(template)
#     dataset /= np.max(dataset)
#
#     # First cross-correlation by initial template.
#     dataset_ccf = []
#     for spec in dataset:
#         dataset_ccf.append(np.correlate(spec, template, mode='same'))
#         # plt.plot(dataset_ccf[-1])
#         # print(dataset_ccf[-1].shape)
#         # plt.show()
#     dataset_ccf = np.asarray(dataset_ccf)
#
#     # Find initial RVs of each CCF.
#     rvs = extract_rvs_from_ccf(pixels, dataset_ccf, width=width)
#
#     return rvs
#
# def extract_rvs_from_ccf(wavelengths_ccf, dataset_ccf, width=5, draw=False):
#     # Find initial RVs of each CCF.
#     rvs = []
#     for i, ccf in enumerate(dataset_ccf):
#         ccf_quad = ccf[np.argmax(ccf) - width: np.argmax(ccf) + width].copy()
#         wavelengths_quad = wavelengths_ccf[np.argmax(ccf) - width: np.argmax(ccf) + width].copy()
#
#         # Fit parabola/quadratic.
#         try:
#             popt, pcov = curve_fit(
#                 amp_gaussian,
#                 wavelengths_quad, ccf_quad,
#                 p0=[np.max(ccf_quad), wavelengths_quad[np.argmax(ccf_quad)], np.median(wavelengths_quad)])
#         except ValueError as err:
#             raise ValueError('Parabola fit region extends beyond dataset. '
#                              'Need wider data or narrower template.')
#
#         # rvs.append(-popt[1] / (2 * popt[0]))
#         rvs.append(popt[1])
#
#         if draw:
#             fig = plt.figure(figsize=(10, 6))
#             ax1 = fig.add_subplot(1, 1, 1)
#
#             ax1.plot(wavelengths_ccf, ccf)
#             ax1.plot(wavelengths_quad, ccf_quad)
#             hr_wavelengths_quad = np.linspace(np.min(wavelengths_quad), np.max(wavelengths_quad), 1000)
#             ax1.plot(hr_wavelengths_quad, amp_gaussian(hr_wavelengths_quad, popt[0], popt[1], popt[2]))
#             ax1.axvline(rvs[-1])
#             # ax1.set_xlim(initial_wavelengths.min(), initial_wavelengths.max())
#
#             plt.tight_layout()
#             plt.show()
#
#     return np.asarray(rvs)
#
# def extract_global_opt(all_data, all_var):
#     # Subtract background.
#     # Todo: poly fit and outlier iterate.
#     left_bkgs = np.mean(all_data[:, :, 0:20], axis=(1, 2))
#     right_bkgs = np.mean(all_data[:, :, -20:], axis=(1, 2))
#     bkg = (left_bkgs + right_bkgs) / 2.
#     all_data = all_data - bkg[:, np.newaxis, np.newaxis]
#
#     x_shifts = np.zeros(all_data.shape[0])
#     y_shifts = np.zeros(all_data.shape[0])
#
#     # R, C = np.meshgrid(row_pixel_vals, col_pixel_vals)
#     while True:
#         # Re-grid.
#         regrid_all_data = np.copy(all_data)
#         regrid_all_variances = np.copy(all_var)
#         # Todo: only regrid from all data into the strip required.
#         for i in range(n_ints):
#             # points = np.stack([np.ravel(R), np.ravel(C)], axis=1)
#             # point_values = np.ravel(all_data[i, :, :])
#             # point_variances = np.ravel(all_var[i, :, :])
#             # new_points = np.stack([R.ravel() - 10, C.ravel() - 0], axis=1)
#             # interp_data = CloughTocher2DInterpolator(points, point_values)
#             # interp_variance = CloughTocher2DInterpolator(points, point_variances)
#             # regrid_all_data[i, :, :] = interp_data(new_points).reshape(n_rows, n_cols)
#             # regrid_all_variances[i, :, :] = interp_var(new_points).reshape(n_rows, n_cols)
#
#             interp_data = interpolate.RectBivariateSpline(row_pixel_vals, col_pixel_vals, all_data[i, :, :],
#                                                           kx=3, ky=3)
#             interp_variance = interpolate.RectBivariateSpline(row_pixel_vals, col_pixel_vals, all_var[i, :, :],
#                                                               kx=3, ky=3)
#             regrid_all_data[i, :, :] = interp_data(row_pixel_vals + y_shifts[i], col_pixel_vals + x_shifts[i])
#             regrid_all_variances[i, :, :] = interp_variance(row_pixel_vals + y_shifts[i],
#                                                             col_pixel_vals + x_shifts[i])
#
#         # Stack.
#         # plt.imshow(regrid_all_data[0])
#         # plt.show()
#         stacked_data = np.sum(regrid_all_data, axis=0)
#         stacked_var = np.sum(regrid_all_variances, axis=0)
#
#         # Build weights map.
#         centre = 34
#         strip_width = 10
#         stacked_d_strip = stacked_data[:, round(centre - strip_width):round(centre + strip_width)]
#         stacked_var_strip = stacked_var[:, round(centre - strip_width):round(centre + strip_width)]
#         p_spec, p_psf = build_global_weightings_img(stacked_d_strip, stacked_var_strip)
#
#         # Extract opt spec and psf.
#         opt_specs = []
#         opt_psfs = []
#         for i in range(n_ints):
#             d_strip = regrid_all_data[i, :, round(centre - strip_width):round(centre + strip_width)]
#             f = np.sum(d_strip, axis=1)
#             f2 = np.sum(d_strip, axis=0)
#
#             v = readout_noise ** 2 + np.abs(f[:, np.newaxis] * p_spec + bkg[i]) / Q
#             opt_specs.append(np.sum(p_spec * d_strip / v, axis=1) / np.sum(p_spec ** 2 / v, axis=1))
#
#             v2 = readout_noise ** 2 + np.abs(f2[np.newaxis, :] * p_psf + bkg[i]) / Q
#             opt_psfs.append(np.sum(p_psf * d_strip / v2, axis=0) / np.sum(p_psf ** 2 / v2, axis=0))
#
#         # Compute shifts.
#         template_spec = np.median(opt_specs, axis=0)
#         template_psf = np.median(opt_psfs, axis=0)
#
#         pixels_spec = row_pixel_vals
#         cc = cross_correlation(pixels_spec, template_spec, np.array(opt_specs), width=5)
#         y_shift_corr = cc - cc[int(n_ints / 2)]
#         y_shifts += y_shift_corr
#
#         pixels_psf = col_pixel_vals[round(centre - strip_width): round(centre + strip_width)]
#         cc = cross_correlation(pixels_psf, template_psf, np.array(opt_psfs), width=5)
#         x_shift_corr = cc - cc[int(n_ints / 2)]
#         x_shifts += x_shift_corr
#
#         print('Shift corr x median: ', np.median(x_shift_corr),
#               'Shift corr y median: ', np.median(y_shift_corr))
#         if np.all(np.abs(y_shift_corr) < 1e-5) and np.all(np.abs(x_shift_corr) < 1e-5):
#             break
#
#     return np.array(opt_specs)
#
# def extract_opt(data_2d, variance_2d):
#     psf_loc = 34
#     strip_width = 10
#     d_strip = data_2d[:, round(psf_loc - strip_width):round(psf_loc + strip_width)]
#     var_strip = variance_2d[:, round(psf_loc - strip_width):round(psf_loc + strip_width)]
#
#     bkg = np.mean(np.append(data_2d[:, 0:20].ravel(), data_2d[:, -20:].ravel()))
#     d = d_strip - bkg
#     f = np.sum(d, axis=1)
#     p_initial = np.copy(d)
#     # p_initial = d / f[:, np.newaxis]
#     # poly_fit_weights = 1 / (var_strip / f[:, np.newaxis] ** 2) ** 0.5
#     poly_fit_weights = 1 / var_strip ** 0.5
#
#     p = []
#     for col in col_pixel_vals[:strip_width * 2]:
#         p_iter = np.copy(p_initial)
#         p_mask = np.ones(row_pixel_vals.shape)
#         while True:
#             for idx in np.where(p_mask == 0)[0]:
#                 p_iter[idx] = np.median(p_iter[np.max((0, idx - 10)): idx + 11])
#             p_coeff = np.polyfit(row_pixel_vals, p_iter[:, col], 10, w=poly_fit_weights[:, col])
#             p_model = np.polyval(p_coeff, row_pixel_vals)
#             residuals = p_iter[:, col] - p_model
#             # plt.plot(p_iter[:, col])
#             # plt.plot(p_model)
#             # plt.show()
#             deviations = np.abs(residuals) / np.std(residuals)
#             max_deviation_idx = np.argmax(deviations)
#             if deviations[max_deviation_idx] > 7.:
#                 p_mask[max_deviation_idx] = 0
#                 continue
#             else:
#                 p.append(p_model)
#                 break
#
#     p = np.array(p).T
#     p[p < 0] = 0.
#     p = p / np.sum(p, axis=1)[:, np.newaxis]
#     v = readout_noise ** 2 + np.abs(f[:, np.newaxis] * p + bkg) / Q
#
#     opt_spec = np.sum(p * d / v, axis=1) / np.sum(p ** 2 / v, axis=1)
#     return opt_spec
#
# def extract_box(data_2d, variance_2d):
#     box_width = 10
#     psf_loc = 34
#     bkg = np.mean(np.append(data_2d[:, 0:20].ravel(), data_2d[:, -20:].ravel()))
#     box_data_2d = data_2d - bkg
#     box_spec = np.sum(box_data_2d[:, round(psf_loc - box_width):round(psf_loc + box_width)], axis=1)
#     return box_spec
#
# ds = []
# vs = []
# for i in range(n_ints):
#     d, v = make_integration_frame(spec_func_means[i], psf_locs[i], draw=False)
#     ds.append(d)
#     vs.append(v)
#
# specs_box = []
# specs_opt = []
# for i in range(n_ints):
#     specs_box.append(extract_box(ds[i], vs[i]))
#     specs_opt.append(extract_opt(ds[i], vs[i]))
# specs_g_opt = extract_global_opt(np.array(ds), np.array(vs))
#
# lcs_box = []
# lcs_opt = []
# lcs_g_opt = []
# specs_box = np.array(specs_box)
# specs_opt = np.array(specs_opt)
# specs_g_opt = np.array(specs_g_opt)
# for i in range(n_bins):
#     if i == n_bins:
#         break
#     lcs_box.append(np.sum(specs_box[:, bins[i]:bins[i + 1]], axis=1))
#     lcs_opt.append(np.sum(specs_opt[:, bins[i]:bins[i + 1]], axis=1))
#     lcs_g_opt.append(np.sum(specs_g_opt[:, bins[i]:bins[i + 1]], axis=1))
#
# for lc_box, lc_opt, lc_g_opt in zip(lcs_box, lcs_opt, lcs_g_opt):
#     plt.scatter(np.arange(0, n_ints, 1), lc_box / np.median(lc_box))
#     plt.scatter(np.arange(0, n_ints, 1), lc_opt / np.median(lc_opt))
#     plt.scatter(np.arange(0, n_ints, 1), lc_g_opt / np.median(lc_g_opt))
#     print(np.std(lc_box / np.median(lc_box)),
#           np.std(lc_opt / np.median(lc_opt)),
#           np.std(lc_g_opt / np.median(lc_g_opt)))
#     plt.ylim(0.95, 1.05)
#     plt.show()
#
#
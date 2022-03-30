import numpy as np
from scipy import stats
from jwst import datamodels
from jwst.stpipe import Step
import matplotlib.pyplot as plt


class Extract1dStep(Step):
    """ Extract 1d spectra.

    A step that will sky subtract, trace find, and extract 1d spectra
    from the 2d rate images using various algorithms.

    """

    spec = """
    bkg_algo = option("constant", "polynomial", default="polynomial")  # background algorithm
    bkg_region = int_list(default=None)  # background region, start, stop, start, stop
    bkg_poly_order = integer(default=1)  # order of polynomial for background fitting
    bkg_smoothing_length = integer(default=None)  # median smooth values over pixel length
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
            sci_model.data = sci_model.data * sci_model.meta.exposure.integration_time
            sci_model.err = sci_model.err * sci_model.meta.exposure.integration_time
            sci_model.meta.bunit_data = 'DN'

            # Background/sky subtract science data.
            sci_data_sub = self.compute_bkg_subtracted_data(
                sci_model.data, sci_model.err)
            if sci_data_sub is None:
                sci_model.meta.cal_step.extract_1d = 'SKIPPED'
                return sci_model
            return

            # Extract 1d spectra.
            spectra = self.extract_1d_spectra(
                D=sci_data_sub,
                V=sci_model.err,
                V_0=read_noise_data,
                Q=gain_data)

            # TODO: build output data type.

            # Update meta.
            sci_model.meta.cal_step.extract_1d = 'COMPLETE'
            sci_model.meta.filetype = '1d spectrum'

        return

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
        """ Compute background/sky subtracted science data. """
        # Cutout data in specified background region.
        sci_data_sub = np.zeros(sci_data.shape)
        all_cols = np.arange(0, sci_data.shape[2])
        bkg_cols = np.r_[self.bkg_region[0]:self.bkg_region[1],
                         self.bkg_region[2]:self.bkg_region[3]]
        bkg_data = sci_data[:, :, bkg_cols]

        # Iterate integrations.
        self.log.info('Computing background.')
        for idx_int, integration in enumerate(bkg_data):

            if self.bkg_algo == 'constant':
                bkg = np.mean(stats.sigmaclip(
                    integration, low=5.0, high=5.0)[0])
                sci_data_sub[idx_int, :, :] -= bkg

            elif self.bkg_algo == 'polynomial':
                bkg = []
                for idx_row, int_row in enumerate(integration):
                    p_coeff = np.polyfit(
                        bkg_cols, int_row, self.bkg_poly_order,
                        w=1/sci_err[idx_int, idx_row, bkg_cols])
                    bkg.append(np.polyval(p_coeff, all_cols))

                    if draw:
                        self._draw_bkg_poly_fits(
                            bkg_cols, int_row, all_cols,
                            np.polyval(p_coeff, all_cols),
                            idx_row)

                bkg = np.array(bkg)
                if self.bkg_smoothing_length is not None:
                    bkg = self._median_smooth_per_column(bkg)
                sci_data_sub[idx_int, :, :] -= bkg

            else:
                self.log.error('Background algorithm not supported. see '
                               'bkg_algo = option("constant", "polynomia'
                               'l", default="polynomial")')
                return None

        return sci_data_sub

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

    def _draw_bkg_poly_fits(self, x_data, y_data, x_model, y_model, idx_row):
        """ Draw the polynomial fits to the background. """
        fig = plt.figure(figsize=(8, 7))
        ax1 = fig.add_subplot(111)
        ax1.scatter(x_data, y_data, s=10, c='#000000',
                    label='Bkg pixels, row={}.'.format(idx_row))
        ax1.plot(x_model, y_model, c='#bc5090',
                 label='Poly fit, order={}.'.format(self.bkg_poly_order))
        ax1.set_xlabel('Pixel row')
        ax1.set_ylabel('DN/s')
        plt.tight_layout()
        plt.show()

    def extract_1d_spectra(self, D, V, V_0, Q):
        """ Extract 1d spectra from bkg subtracted science data. """
        # Potentially we want to cut a border away from the edge too.
        # Then on shift-reg-grid we cut to the data strip we care about.

        # Option 1.
        # Box.

        # Option 2.
        # Optimal extraction per frame.

        # Option 3.
        # Global optimal extraction.
        return []

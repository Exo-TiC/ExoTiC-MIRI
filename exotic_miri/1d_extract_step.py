from jwst import datamodels
from jwst.stpipe import Step

from jwst.pipeline import calwebb_spec2


class Extract1dStep(Step):
    """ Extract a 1-d spectrum from 2-d data.

    Blurb.

    """

    spec = """
    extract_algo = option("box", "optimal", default="box")  # extraction algorithm
    """

    reference_file_types = ['extract1d', 'apcorr', 'wavemap', 'spectrace', 'specprofile']

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model

        Returns
        -------
        JWST data model
            This will be `input_model` if the step was skipped; otherwise,
            it will be a model containing 1-D extracted spectra.
        """

        # Open the input and figure out what type of model it is
        input_model = datamodels.open(input)

        # Check input model type.
        if not isinstance(input_model, datamodels.CubeModel):
            self.log.error(f'Input is a {str(type(input_model))}, ')
            self.log.error('which was not expected for extract_1d')
            self.log.error('extract_1d will be skipped.')
            input_model.meta.cal_step.extract_1d = 'SKIPPED'
            return input_model

        # Data observation mode.
        if not input_model.meta.exposure.type == 'MIR_LRS-SLITLESS':
            self.log.error(f'Observation is a {input_model.meta.exposure.type}, ')
            self.log.error('which is not supported by ExoTic-MIRIs extract_1d')
            self.log.error('extract_1d will be skipped.')
            input_model.meta.cal_step.extract_1d = 'SKIPPED'
            return input_model

        # TODO: see what this does. We will need the read noise
        self.get_reference_file(input_model, 'apcorr')

        # TODO: do extraction. diff types
        # TODO: Need to add iterations for not using statistical outliers in weighting map.

        # TODO: build output data type.

        result = extract.run_extract1d(
            input_model,
            extract_ref,
            apcorr_ref,
            self.smoothing_length,
            self.bkg_fit,
            self.bkg_order,
            self.bkg_sigma_clip,
            self.log_increment,
            self.subtract_background,
            self.use_source_posn,
            self.center_xy,
            was_source_model=False,
        )

        # Set the step flag to complete
        result.meta.cal_step.extract_1d = 'COMPLETE'
        result.meta.filetype = '1d spectrum'

        input_model.close()

        return

    def stuff(self):
        # Todo -- DN vs e- vs rates for input data, func of 2d.
        # Todo -- Sky background subtraction in 2d.
        # Todo -- non-parametric vs polynomial fits to background and weightings model.
        # Todo -- convergence criterion: all or median < eps.
        # Todo -- make a notebook example of the three methods.
        # Todo -- how can we do it for all integrations, or just per data chunk?
        # Todo -- implement as a step.
        return

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




from jwst import datamodels
from jwst.stpipe import Step


class Extract1dStep(Step):
    """Extract a 1-d spectrum from 2-d data

    Blurb.

    """

    spec = """
    smoothing_length = integer(default=None)  # background smoothing size
    bkg_fit = option("poly", "mean", "median", default="poly")  # background fitting type
    bkg_order = integer(default=None, min=0)  # order of background polynomial fit
    bkg_sigma_clip = float(default=3.0)  # background sigma clipping threshold
    log_increment = integer(default=50)  # increment for multi-integration log messages
    subtract_background = boolean(default=None)  # subtract background?
    """

    reference_file_types = ['extract1d', 'apcorr', 'wavemap', 'spectrace', 'specprofile', 'speckernel']

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
        if not input_model.meta.exposure.type == 'MIRI_LRS':
            self.log.error(f'Observation is a {input_model.meta.exposure.type}, ')
            self.log.error('which is not supported by ExoTic-MIRIs extract_1d')
            self.log.error('extract_1d will be skipped.')
            input_model.meta.cal_step.extract_1d = 'SKIPPED'
            return input_model


        # TODO: see what this does.
        self.get_reference_file(input_model, 'apcorr')

        # TODO: do extraction.

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

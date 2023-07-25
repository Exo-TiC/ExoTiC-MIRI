import os
import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step


class GetDefaultReadNoise(Step):
    """ Get the default readnoise.

    This enables the user to get and save readnoise data from
    the default CRDS files.

    """

    spec = """
    data_seg_name = string(default=None)  # any data segment name.
    stage_1_dir = string(default=None)  # directory of stage 1 products.
    stage_2_dir = string(default=None)  # directory of stage 2 products.
    trim_col_start = integer(default=0)  # trim columns before this index.
    trim_col_end = integer(default=73)  # trim columns on and after this index.
    gain_value = float(default=3.1)  # gain value to convert form DN to electrons.
    median_value = boolean(default=False)  # only return median value.
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type CubeModel.

        Returns
        -------
        array or float
            Readnoise array [electrons], or float if median_value=True.

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error('Input is a {} which was not expected for '
                               'ReadNoiseStep, skipping step.'.format(
                                str(type(input_model))))
                return input_model

            # Data subarray positions.
            self.log.info('Getting subarray position.')
            stage_1_data_chunk = os.path.join(
                self.stage_1_dir, '{}_stage_1.fits'.format(self.data_seg_name))
            data_header = fits.getheader(stage_1_data_chunk)
            xstart = data_header['SUBSTRT1']
            ystart = data_header['SUBSTRT2']
            nx = data_header['SUBSIZE1']
            ny = data_header['SUBSIZE2']
            n_groups = data_header[""]

            # Extract readnoise.
            readnoise_filename = self.get_reference_file(input_model, 'readnoise')
            readnoise_header = fits.getheader(readnoise_filename)
            xstart_readnoise = readnoise_header['SUBSTRT1']
            ystart_readnoise = readnoise_header['SUBSTRT2']
            ystart_trim = ystart - ystart_readnoise + 1
            xstart_trim = xstart - xstart_readnoise + 1
            self.log.info('Getting readnoise.')
            with datamodels.ReadnoiseModel(readnoise_filename) as readnoise_model:
                readnoise_model.data = readnoise_model.data[
                    ystart_trim:ystart_trim + ny,
                    xstart_trim:xstart_trim + nx].copy()
                readnoise_model.data = readnoise_model.data[
                    :, self.trim_col_start:self.trim_col_end].copy()

                # Convert from DN to electrons.
                readnoise_model.data *= self.gain_value / np.sqrt(2. * nframes)

                # Save.
                readnoise_model.save(path=os.path.join(
                    self.stage_2_dir, '{}_stage_2_readnoise.fits'.format(
                        self.data_seg_name)))

                # Median value.
                med_read_noise = np.median(readnoise_model.data)
                self.log.info('Median readnoise={} electrons.'.format(med_read_noise))

                if self.median_value:
                    return float(med_read_noise)
                else:
                    return readnoise_model.data

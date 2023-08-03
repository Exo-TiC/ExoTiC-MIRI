import os
import numpy as np
from astropy.io import fits
from jwst import datamodels
from jwst.stpipe import Step


class GetWavelengthMap(Step):
    """ Get the wavelength map. """

    spec = """
    trim_col_start = integer(default=0)  # trim columns before this index.
    trim_col_end = integer(default=72)  # trim columns on and after this index.
    save = boolean(default=False)  # save map to disk as .fits.
    save_path = string(default=None)  # save map path.
    """

    def process(self, input):
        """ Get and save the wavelength map data. This is the mapping from
        the detector pixels (row_idx, col_idx) to wavelength (lambda). To
        run this step the input must first have had the
        jwst.calwebb_spec2.assign_wcs_step
        and jwst.calwebb_spec2.srctype_step already run.

        Parameters
        ----------
        input: jwst.datamodels.CubeModel
            This is an rateints.fits loaded data segment.

        Returns
        -------
        wavelength_map: np.ndarray
            The wavelength map with shape (n_rows, n_cols).

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error("Input is a {} which was not expected for "
                               "WavelengthMapStep, skipping step.".format(
                                str(type(input_model))))
                return input_model

            # Extract wavelength map.
            self.log.info("Getting wavelength map.")
            row_g, col_g = np.mgrid[0:input_model.data.shape[1],
                                    0:input_model.data.shape[2]]
            wavelength_map = input_model.meta.wcs(
                col_g.ravel(), row_g.ravel())[-1].reshape(
                input_model.data.shape[1:])
            wavelength_map = wavelength_map[
                :, self.trim_col_start:self.trim_col_end]

            if self.save:
                hdu = fits.PrimaryHDU(wavelength_map)
                hdul = fits.HDUList([hdu])
                wave_map_name = "{}_wavelengthmap.fits".format(
                    self.data_chunk_name)
                hdul.writeto(os.path.join(
                    self.save_path, wave_map_name), overwrite=True)

        return wavelength_map

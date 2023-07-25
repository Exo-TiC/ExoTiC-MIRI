import os
import shutil
from jwst import datamodels
from jwst.stpipe import Step
from jwst.pipeline.calwebb_spec2 import flat_field_step


class GetDefaultFlatField(Step):
    """ Get the default flat field.

    This enables the user to get and save flat field data from
    the default CRDS files.

    """

    spec = """
    data_seg_name = string(default=None)  # any data segment name.
    stage_2_dir = string(default=None)  # directory of stage 2 products.
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type CubeModel.

        Returns
        -------
        JWST data model
            The flat field model.

        """
        with datamodels.open(input) as input_model:

            # Check input model type.
            if not isinstance(input_model, datamodels.CubeModel):
                self.log.error('Input is a {} which was not expected for '
                               'FlatFieldStep, skipping step.'.format(
                                str(type(input_model))))
                return input_model

            # Using stsci flats step.
            stsci_flat_field = flat_field_step.FlatFieldStep()
            output_model = stsci_flat_field.call(input_model, save_interpolated_flat=True)

            # Save.
            flat_name = '{}_stage_1_{}.fits'.format(
                self.data_seg_name, stsci_flat_field.flat_suffix)
            flat_name_new = '{}_stage_2_{}.fits'.format(
                self.data_seg_name, stsci_flat_field.flat_suffix)
            shutil.move(flat_name, os.path.join(self.stage_2_dir, flat_name_new))

            return output_model

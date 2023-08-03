import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class DropGroupsStep(Step):
    """ Drop groups step. """

    spec = """
    drop_groups = int_list(default=None)  # groups to drop, zero-indexed.
    """

    def process(self, input):
        """ Drop groups which may be adversely affecting the ramps. This
        may be due to detector effects such RSCD and/or the last frame effect.
        This step simply marks groups as do_not_use and are thus ignored
        in subsequent processing, such as by jwst.calwebb_detector1.ramp_fit_step.

        Parameters
        ----------
        input: jwst.datamodels.RampModel
            This is an uncal.fits loaded data segment.
        drop_groups: list of integers
            These integers are the groups to be dropped. The integers are
            zero-indexed such that 0 is the first group.

        Returns
        -------
        output: jwst.datamodels.RampModel
            A RampModel with groupdq flags set as do_not_use (2**0).

        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            thinned_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel):
                self.log.error("Input is a {} which was not expected for "
                               "drop_groups_step, skipping step.".format(
                                str(type(input_model))))
                thinned_model.meta.cal_step.drop_groups = "SKIPPED"
                return thinned_model

            # Check the observation mode.
            if not input_model.meta.exposure.type == "MIR_LRS-SLITLESS":
                self.log.error("Observation is a {} which is not supported "
                               "by ExoTic-MIRIs drop_groups_step, skipping "
                               "step.".format(input_model.meta.exposure.type))
                thinned_model.meta.cal_step.drop_groups = "SKIPPED"
                return thinned_model

            # Check groups to drop exist.
            min_g_drop = np.min(self.drop_groups)
            max_g_drop = np.max(self.drop_groups)
            current_n_groups = thinned_model.meta.exposure.ngroups
            if min_g_drop < 0 or max_g_drop > current_n_groups - 1:
                self.log.error("Not all groups listed for dropping exist, req"
                               "uested to drop groups between {} and {} when "
                               "current groups only span 0 to {}. Check your "
                               "input list is zero indexed, skipping step."
                               .format(min_g_drop, max_g_drop,
                                       current_n_groups - 1))
                thinned_model.meta.cal_step.drop_groups = "SKIPPED"
                return thinned_model

            # Set dropped groups as do not use 2**0.
            thinned_model.groupdq[:, self.drop_groups, :, :] = np.bitwise_or(
                thinned_model.groupdq[:, self.drop_groups, :, :], 2**0)

            # Update meta.
            thinned_model.meta.cal_step.drop_groups = "COMPLETE"

        return thinned_model

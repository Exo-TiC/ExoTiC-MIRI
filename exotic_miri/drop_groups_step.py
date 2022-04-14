import numpy as np
from jwst import datamodels
from jwst.stpipe import Step


class DropGroupsStep(Step):
    """ Drop groups within integrations.

    This steps enables the user to drop groups from each integration
    which may be adversely affecting the ramps.

    """

    spec = """
    drop_groups = int_list(default=None)  # groups to drop, zero-indexed.
    """

    def process(self, input):
        """Execute the step.

        Parameters
        ----------
        input: JWST data model
            A data model of type RampModel.

        Returns
        -------
        JWST data model
            A RampModel with updated groups, unless the
            step is skipped in which case `input_model` is returned.
        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            thinned_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel):
                self.log.error('Input is a {} which was not expected for '
                               'drop_groups_step, skipping step.'.format(
                                str(type(input_model))))
                thinned_model.meta.cal_step.drop_groups = 'SKIPPED'
                return thinned_model

            # Check the observation mode.
            if not input_model.meta.exposure.type == 'MIR_LRS-SLITLESS':
                self.log.error('Observation is a {} which is not supported '
                               'by ExoTic-MIRIs drop_groups_step, skipping '
                               'step.'.format(input_model.meta.exposure.type))
                thinned_model.meta.cal_step.drop_groups = 'SKIPPED'
                return thinned_model

            # Check groups to drop exist.
            min_g_drop = np.min(self.drop_groups)
            max_g_drop = np.max(self.drop_groups)
            current_n_groups = thinned_model.meta.exposure.ngroups
            if min_g_drop < 0 or max_g_drop > current_n_groups - 1:
                self.log.error('Not all groups listed for dropping exist, req'
                               'uested to drop groups between {} and {} when '
                               'current groups only span 0 to {}. Check your '
                               'input list is zero indexed, skipping step.'
                               .format(min_g_drop, max_g_drop,
                                       current_n_groups - 1))
                thinned_model.meta.cal_step.drop_groups = 'SKIPPED'
                return thinned_model

            # Compute wanted groups.
            current_groups = np.arange(0, current_n_groups, 1)
            wanted_groups = current_groups[~np.isin(current_groups, self.drop_groups)]

            # Drop groups.
            thinned_model.data = thinned_model.data[:, wanted_groups, :, :]
            thinned_model.err = thinned_model.err[:, wanted_groups, :, :]
            thinned_model.groupdq = thinned_model.groupdq[:, wanted_groups, :, :]

            # Update meta.
            thinned_model.meta.ngroups = thinned_model.data.shape[1]
            thinned_model.meta.ngroups_file = thinned_model.data.shape[1]
            span_g_wanted = np.max(wanted_groups) + 1 - np.min(wanted_groups)
            span_decrease_f = span_g_wanted / current_n_groups
            thinned_model.meta.exposure.integration_time = \
                thinned_model.meta.exposure.integration_time * span_decrease_f
            thinned_model.meta.exposure.ngroups = thinned_model.data.shape[1]
            if thinned_model._shape:
                thinned_model._shape = thinned_model.data.shape
            thinned_model.meta.cal_step.drop_groups = 'COMPLETE'

        return thinned_model

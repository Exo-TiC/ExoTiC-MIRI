from jwst import datamodels
from jwst.stpipe import Step


class RegroupStep(Step):
    """ Regroup groups into integrations.

    This steps enables the user to regroup integrations, comprised
    of n groups, into several smaller integrations, comprised of m
    groups, where n is a multiple of m.

    """

    spec = """
    n_groups = integer(default=10)  # new number of groups per integration
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
            A RampModel with updated integration groupings, unless the
            step is skipped in which case `input_model` is returned.
        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            regrouped_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel):
                self.log.error('Input is a {} which was not expected for '
                               'regroup_step, skipping step.'.format(
                                str(type(input_model))))
                regrouped_model.meta.cal_step.regroup = 'SKIPPED'
                return regrouped_model

            # Check the observation mode.
            if not input_model.meta.exposure.type == 'MIR_LRS-SLITLESS':
                self.log.error('Observation is a {} which is not supported '
                               'by ExoTic-MIRIs regroup_step, skipping step.'
                               .format(input_model.meta.exposure.type))
                regrouped_model.meta.cal_step.regroup = 'SKIPPED'
                return regrouped_model

            # Check original number of groups is a multiple of n_groups.
            n = regrouped_model.meta.exposure.ngroups
            if not n % self.n_groups == 0:
                self.log.error('Regrouping to {} groups is not possible for '
                               'the original group number {}. It must be a '
                               'multiple, skipping step.'.format(
                                self.n_groups, n))
                regrouped_model.meta.cal_step.regroup = 'SKIPPED'
                return regrouped_model

            # Compute change in integration sizes.
            d_factor = self.n_groups / regrouped_model.data.shape[1]
            n_int = int(regrouped_model.data.shape[0] / d_factor)

            # Regroup data.
            regrouped_model.data = regrouped_model.data.reshape(
                n_int, self.n_groups, 416, 72)
            regrouped_model.err = regrouped_model.err.reshape(
                n_int, self.n_groups, 416, 72)
            regrouped_model.groupdq = regrouped_model.groupdq.reshape(
                n_int, self.n_groups, 416, 72)

            # Update meta.
            regrouped_model.meta.ngroups = self.n_groups
            regrouped_model.meta.ngroups_file = self.n_groups
            regrouped_model.meta.nints = n_int
            regrouped_model.meta.nints_file = n_int
            regrouped_model.meta.exposure.integration_time = \
                regrouped_model.meta.exposure.integration_time * d_factor
            regrouped_model.meta.exposure.ngroups = self.n_groups
            regrouped_model.meta.exposure.nints = n_int
            if regrouped_model._shape:
                regrouped_model._shape = regrouped_model.data.shape
            regrouped_model.meta.cal_step.regroup = 'COMPLETE'

        return regrouped_model

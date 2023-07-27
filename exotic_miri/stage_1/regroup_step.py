from jwst import datamodels
from jwst.stpipe import Step


class RegroupStep(Step):
    """ Regroup long integrations into more shorter integrations. """

    spec = """
    n_groups = integer(default=10)  # new number of groups per integration
    """

    def process(self, input):
        """ Regroup integrations, comprised of m groups, into several smaller
        integrations, comprised of n groups, where m is a multiple of n. This
        may not be helpful but it is here if you want to try it.
        TODO: add support for non-multiples.

        Parameters
        ----------
        input: jwst.datamodels.RampModel
            This is an uncal.fits loaded data segment.
        n_groups: integer
            The new number of groups per integration.

        Returns
        -------
        output: jwst.datamodels.RampModel
            A RampModel reshaped into n_groups.

        """
        with datamodels.open(input) as input_model:

            # Copy input model.
            regrouped_model = input_model.copy()

            # Check input model type.
            if not isinstance(input_model, datamodels.RampModel):
                self.log.error("Input is a {} which was not expected for "
                               "regroup_step, skipping step.".format(
                                str(type(input_model))))
                regrouped_model.meta.cal_step.regroup = "SKIPPED"
                return regrouped_model

            # Check the observation mode.
            if not input_model.meta.exposure.type == "MIR_LRS-SLITLESS":
                self.log.error("Observation is a {} which is not supported "
                               "by ExoTic-MIRIs regroup_step, skipping step."
                               .format(input_model.meta.exposure.type))
                regrouped_model.meta.cal_step.regroup = "SKIPPED"
                return regrouped_model

            # Check original number of groups is a multiple of n_groups.
            n = regrouped_model.meta.exposure.ngroups
            if not n % self.n_groups == 0:
                self.log.error("Regrouping to {} groups is not possible for "
                               "the original group number {}. It must be a "
                               "multiple, skipping step.".format(
                                self.n_groups, n))
                regrouped_model.meta.cal_step.regroup = "SKIPPED"
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
            regrouped_model.meta.cal_step.regroup = "COMPLETE"

        return regrouped_model

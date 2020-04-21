import autofit as af


class Result(af.Result):
    def __init__(
        self,
        instance,
        log_likelihood,
        previous_model,
        gaussian_tuples,
        analysis,
        optimizer,
    ):
        """
        The result of a phase
        """
        super().__init__(
            instance=instance,
            log_likelihood=log_likelihood,
            previous_model=previous_model,
            gaussian_tuples=gaussian_tuples,
        )

        self.analysis = analysis
        self.optimizer = optimizer

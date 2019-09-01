import math
import os

import autofit as af
from test.integration import integration_util
from test.simulation import simulation_util


class MockNLO(af.NonLinearOptimizer):
    def fit(self, analysis):
        if self.variable.prior_count == 0:
            raise AssertionError("There are no priors associated with the variable!")
        if self.variable.prior_count != len(self.variable.unique_prior_paths):
            raise AssertionError(
                "Prior count doesn't match number of unique prior paths"
            )
        index = 0
        unit_vector = self.variable.prior_count * [0.5]
        while True:
            try:
                instance = self.variable.instance_from_unit_vector(unit_vector)
                fit = analysis.fit(instance)
                break
            except af.exc.FitException as e:
                unit_vector[index] += 0.1
                if unit_vector[index] >= 1:
                    raise e
                index = (index + 1) % self.variable.prior_count
        return af.Result(
            instance,
            fit,
            self.variable,
            gaussian_tuples=[
                (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                for prior in sorted(self.variable.priors, key=lambda prior: prior.id)
            ],
        )


def run(
    module,
    test_name=None,
    optimizer_class=af.MultiNest,
    config_folder="config",
    cti_settings=None,
    load_cosmic_ray_image=False,
    pool=None,
):
    test_name = test_name or module.test_name
    test_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))
    output_path = test_path + "output/"
    config_path = test_path + config_folder
    af.conf.instance = af.conf.Config(config_path=config_path, output_path=output_path)
    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    data = list(
        map(
            lambda normalization: simulation_util.load_test_ci_data(
                ci_data_type=module.ci_data_type,
                ci_data_model=module.ci_data_model,
                ci_data_resolution=module.ci_data_resolution,
                normalization=normalization,
                load_cosmic_ray_image=load_cosmic_ray_image,
            ),
            module.ci_normalizations,
        )
    )

    module.make_pipeline(
        name=test_name,
        phase_folders=[module.test_type, test_name],
        optimizer_class=optimizer_class,
    ).run(ci_datas=data, cti_settings=cti_settings, pool=pool)


def run_a_mock(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_mock",
        optimizer_class=MockNLO,
        config_folder="config_mock",
    )


def run_with_multi_nest(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_nest",
        optimizer_class=af.MultiNest,
        config_folder="config_mock",
    )

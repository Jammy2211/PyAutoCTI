import math
import os

import autofit as af
from test import integration_util
from test import simulate_util


class MockNLO(af.NonLinearOptimizer):
    def fit(self, analysis):
        if self.model.prior_count == 0:
            raise AssertionError("There are no priors associated with the model!")
        if self.model.prior_count != len(self.model.unique_prior_paths):
            raise AssertionError(
                "Prior count doesn't match number of unique prior paths"
            )
        index = 0
        unit_vector = self.model.prior_count * [0.5]
        while True:
            try:
                instance = self.model.instance_from_unit_vector(unit_vector)
                fit = analysis.fit(instance)
                break
            except af.exc.FitException as e:
                unit_vector[index] += 0.1
                if unit_vector[index] >= 1:
                    raise e
                index = (index + 1) % self.model.prior_count
        return af.Result(
            instance,
            fit,
            self.model,
            gaussian_tuples=[
                (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                for prior in sorted(self.model.priors, key=lambda prior: prior.id)
            ],
        )


def run(
    module,
    test_name=None,
    non_linear_class=af.MultiNest,
    config_folder="config",
    cti_settings=None,
    load_cosmic_ray_map=False,
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
            lambda normalization: simulate_util.load_test_ci_data(
                ci_data_type=module.ci_data_type,
                ci_data_model=module.ci_data_model,
                ci_data_resolution=module.ci_data_resolution,
                normalization=normalization,
                load_cosmic_ray_map=load_cosmic_ray_map,
            ),
            module.ci_normalizations,
        )
    )

    module.make_pipeline_no_lens_light(
        name=test_name,
        phase_folders=[module.test_type, test_name],
        non_linear_class=non_linear_class,
    ).run(ci_datas=data, cti_settings=cti_settings, pool=pool)


def run_a_mock(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_mock",
        non_linear_class=MockNLO,
        config_folder="config_mock",
    )


def run_with_multi_nest(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_nest",
        non_linear_class=af.MultiNest,
        config_folder="config_mock",
    )

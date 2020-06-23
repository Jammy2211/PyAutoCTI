import os

import autofit as af
from test_autocti.integration import integration_util
from test_autocti.simulate import simulate_util


def run(
    module,
    test_name=None,
    search=af.DynestyStatic(),
    config_folder="config",
    clocker=None,
    pool=None,
):
    test_name = test_name or module.test_name
    test_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))
    output_path = test_path + "output/"
    config_path = test_path + config_folder
    conf.instance = conf.Config(config_path=config_path, output_path=output_path)
    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    datasets = list(
        map(
            lambda normalization: simulate_util.load_test_ci_data(
                ci_data_type=module.ci_data_type,
                ci_data_model=module.ci_data_model,
                resolution=module.resolution,
                normalization=normalization,
            ),
            module.ci_normalizations,
        )
    )

    pipeline = module.make_pipeline(
        name=test_name, folders=[module.test_type, test_name], search=search
    )

    pipeline.run(datasets=datasets, clocker=clocker, pool=pool)


def run_a_mock(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_mock",
        search=af.MockSearch,
        config_folder="config_mock",
    )


def run_with_multi_nest(module):
    # noinspection PyTypeChecker
    run(
        module,
        test_name=f"{module.test_name}_nest",
        search=af.DynestyStatic(),
        config_folder="config_mock",
    )

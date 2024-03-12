import os
import pytest

import autofit as af
import autocti as ac


def test__save_results__delta_ellipyicity_output_to_json(
    imaging_ci_7x7,
    pre_cti_data_7x7,
    traps_x1,
    ccd,
    parallel_clocker_2d,
):
    analysis = ac.AnalysisCTI(
        dataset=imaging_ci_7x7,
        clocker=parallel_clocker_2d,
        settings_cti=ac.SettingsCTI2D(),
    )

    paths = af.DirectoryPaths()

    model = af.Collection(
        cti=af.Model(
            ac.CTI2D, parallel_trap_list=[ac.TrapInstantCapture], parallel_ccd=ccd
        ),
    )

    parameters = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

    sample_list = af.Sample.from_lists(
        model=model,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0, 3.0],
        log_prior_list=[0.0, 0.0, 0.0],
        weight_list=[0.2, 0.2, 0.2],
    )

    samples = ac.m.MockSamples(sample_list=sample_list, model=model)

    analysis.save_results_combined(
        paths=paths,
        result=ac.m.MockResult(samples=samples, model=model),
    )

    delta_ellipticity = ac.from_json(
        file_path=paths._files_path / "delta_ellipticity.json"
    )

    # Uncomment once CTI build moves to new arctic

    #    assert delta_ellipticity == pytest.approx(-0.403649850, 1.0e-4)

    os.remove(paths._files_path / "delta_ellipticity.json")

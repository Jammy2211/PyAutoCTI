from os import path
import pytest

from autoconf import conf
import autofit as af
import autocti as ac

from test_autocti.aggregator.conftest import clean


def test__fit_imaging_ci_randomly_drawn_via_pdf_gen_from(
    imaging_ci_7x7, parallel_clocker_2d, samples_2d, model_2d
):
    path_prefix = "aggregator_fit_imaging_ci_gen"

    database_file = path.join(conf.instance.output_path, "fit_imaging_ci.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = ac.m.MockSearch(
        samples=samples_2d, result=ac.m.MockResult(model=model_2d, samples=samples_2d)
    )
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)
    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)
    search.fit(model=model_2d, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    fit_imaging_ci_agg = ac.agg.FitImagingCIAgg(aggregator=agg)
    fit_imaging_ci_pdf_gen = fit_imaging_ci_agg.randomly_drawn_via_pdf_gen_from(
        total_samples=2
    )

    i = 0

    for fit_imaging_ci_gen in fit_imaging_ci_pdf_gen:
        for fit_imaging_ci_list in fit_imaging_ci_gen:
            print(fit_imaging_ci_list)
            i += 1

            assert fit_imaging_ci_list[0].post_cti_data[0] == pytest.approx(1.0, 1.0e-4)

    assert i == 2

    clean(database_file=database_file, result_path=result_path)


def test__fit_imaging_ci_all_above_weight_gen(
    imaging_ci_7x7, parallel_clocker_2d, samples_2d, model_2d
):
    path_prefix = "aggregator_fit_imaging_ci_gen"

    database_file = path.join(conf.instance.output_path, "fit_imaging_ci.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = ac.m.MockSearch(
        samples=samples_2d, result=ac.m.MockResult(model=model_2d, samples=samples_2d)
    )
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)
    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)
    search.fit(model=model_2d, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    fit_imaging_ci_agg = ac.agg.FitImagingCIAgg(aggregator=agg)
    fit_imaging_ci_pdf_gen = fit_imaging_ci_agg.all_above_weight_gen_from(
        minimum_weight=-1.0
    )

    i = 0

    for fit_imaging_ci_gen in fit_imaging_ci_pdf_gen:
        for fit_imaging_ci_list in fit_imaging_ci_gen:
            i += 1

    assert i == 2

    clean(database_file=database_file, result_path=result_path)

from os import path
import pytest

from autoconf import conf
import autofit as af
import autocti as ac

from test_autogalaxy.aggregator.conftest import clean


def test__fit_dataset_1d_randomly_drawn_via_pdf_gen_from(
    dataset_1d_7, clocker_1d, samples_1d, model_1d
):
    path_prefix = "aggregator_fit_dataset_1d_gen"

    database_file = path.join(conf.instance.output_path, "fit_dataset_1d.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = ac.m.MockSearch(
        samples=samples_1d, result=ac.m.MockResult(model=model_1d, samples=samples_1d)
    )
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)
    analysis = ac.AnalysisDataset1D(dataset=dataset_1d_7, clocker=clocker_1d)
    search.fit(model=model_1d, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    fit_dataset_1d_agg = ac.agg.FitDataset1DAgg(aggregator=agg)
    fit_dataset_1d_pdf_gen = fit_dataset_1d_agg.randomly_drawn_via_pdf_gen_from(
        total_samples=2
    )

    i = 0

    for fit_dataset_1d_gen in fit_dataset_1d_pdf_gen:
        for fit_dataset_1d in fit_dataset_1d_gen:
            i += 1

            assert fit_dataset_1d.post_cti_data[0] == pytest.approx(1.0, 1.0e-4)

    assert i == 2

    clean(database_file=database_file, result_path=result_path)


def test__fit_dataset_1d_all_above_weight_gen(
    dataset_1d_7, mask_1d_7_unmasked, clocker_1d, samples_1d, model_1d
):
    path_prefix = "aggregator_fit_dataset_1d_gen"

    database_file = path.join(conf.instance.output_path, "fit_dataset_1d.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = ac.m.MockSearch(
        samples=samples_1d, result=ac.m.MockResult(model=model_1d, samples=samples_1d)
    )
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)
    analysis = ac.AnalysisDataset1D(dataset=dataset_1d_7, clocker=clocker_1)
    search.fit(model=model_1d, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    fit_dataset_1d_agg = ac.agg.FitDataset1DAgg(aggregator=agg)
    fit_dataset_1d_pdf_gen = fit_dataset_1d_agg.all_above_weight_gen_from(
        minimum_weight=-1.0
    )

    i = 0

    for fit_dataset_1d_gen in fit_dataset_1d_pdf_gen:
        for fit_dataset_1d in fit_dataset_1d_gen:
            i += 1

            if i == 1:
                assert fit_dataset_1d.post_cti_data[0] == pytest.approx(1.0, 1.0e-4)

            if i == 2:
                assert fit_dataset_1d.post_cti_data[0] == pytest.approx(1.0, 1.0e-4)

    assert i == 2

    clean(database_file=database_file, result_path=result_path)

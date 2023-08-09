import copy
import pytest

import autocti as ac

from test_autocti.aggregator.conftest import clean, aggregator_from

database_file = "db_imaging_ci"

def test__dataset_gen_from__analysis_has_single_dataset(
    imaging_ci_7x7, parallel_clocker_2d, samples_2d, model_2d
):

    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model_2d,
        samples=samples_2d,
    )

    dataset_agg = ac.agg.ImagingCIAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_list_gen_from()

    for dataset_list in dataset_gen:
        assert (dataset_list[0].data == imaging_ci_7x7.data).all()
        assert dataset_list[0].layout.parallel_overscan[1] == pytest.approx(
            imaging_ci_7x7.layout.parallel_overscan[1], 1.0e-4
        )

    clean(database_file=database_file)


def test__dataset_gen_from__analysis_has_multi_dataset(
    imaging_ci_7x7, parallel_clocker_2d, samples_2d, model_2d
):

    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis + analysis,
        model=model_2d,
        samples=samples_2d,
    )

    dataset_agg = ac.agg.ImagingCIAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_list_gen_from()

    for dataset_list in dataset_gen:
        assert (dataset_list[0].data == imaging_ci_7x7.data).all()
        assert (dataset_list[1].data == imaging_ci_7x7.data).all()

        assert dataset_list[0].layout.parallel_overscan[1] == pytest.approx(
            imaging_ci_7x7.layout.parallel_overscan[1], 1.0e-4
        )
        assert dataset_list[1].layout.parallel_overscan[1] == pytest.approx(
            imaging_ci_7x7.layout.parallel_overscan[1], 1.0e-4
        )

    clean(database_file=database_file)

def test__dataset_gen_from__analysis_use_dataset_full(
    imaging_ci_7x7, parallel_clocker_2d, samples_2d, model_2d
):

    imaging_ci_7x7_full = copy.copy(imaging_ci_7x7)
    imaging_ci_7x7_full.data[0] = 100.0

    analysis = ac.AnalysisImagingCI(
        dataset=imaging_ci_7x7,
        clocker=parallel_clocker_2d,
        dataset_full=imaging_ci_7x7_full,
    )

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis + analysis,
        model=model_2d,
        samples=samples_2d,
    )

    dataset_agg = ac.agg.ImagingCIAgg(aggregator=agg, use_dataset_full=True)
    dataset_gen = dataset_agg.dataset_list_gen_from()

    for dataset_list in dataset_gen:
        assert dataset_list[0].data[0] == pytest.approx(100.0, 1.0e-4)
        assert dataset_list[1].data[0] == pytest.approx(100.0, 1.0e-4)

        assert (dataset_list[0].data == imaging_ci_7x7.data).all()
        assert (dataset_list[1].data == imaging_ci_7x7.data).all()

        assert dataset_list[0].layout.parallel_overscan[1] == pytest.approx(
            imaging_ci_7x7.layout.parallel_overscan[1], 1.0e-4
        )
        assert dataset_list[1].layout.parallel_overscan[1] == pytest.approx(
            imaging_ci_7x7.layout.parallel_overscan[1], 1.0e-4
        )

    clean(database_file=database_file)
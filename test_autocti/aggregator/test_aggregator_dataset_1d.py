import copy
import pytest

import autocti as ac

from test_autocti.aggregator.conftest import clean, aggregator_from

database_file = "db_dataset_1d"


def test__dataset_gen_from__analysis_has_single_dataset(
    dataset_1d_7, clocker_1d, samples_1d, model_1d
):
    analysis = ac.AnalysisDataset1D(dataset=dataset_1d_7, clocker=clocker_1d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model_1d,
        samples=samples_1d,
    )

    dataset_agg = ac.agg.Dataset1DAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_list_gen_from()

    for dataset in dataset_gen:
        assert (dataset[0].data == dataset_1d_7.data).all()
        assert dataset[0].layout.prescan[1] == pytest.approx(
            dataset_1d_7.layout.prescan[1], 1.0e-4
        )

    clean(database_file=database_file)


def test__dataset_gen_from__analysis_has_multi_dataset(
    dataset_1d_7, clocker_1d, samples_1d, model_1d
):
    analysis = ac.AnalysisDataset1D(dataset=dataset_1d_7, clocker=clocker_1d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis + analysis,
        model=model_1d,
        samples=samples_1d,
    )

    dataset_agg = ac.agg.Dataset1DAgg(aggregator=agg, use_dataset_full=False)
    dataset_gen = dataset_agg.dataset_list_gen_from()

    for dataset_list in dataset_gen:
        assert (dataset_list[0].data == dataset_1d_7.data).all()
        assert (dataset_list[1].data == dataset_1d_7.data).all()
        assert dataset_list[0].layout.prescan[1] == pytest.approx(
            dataset_1d_7.layout.prescan[1], 1.0e-4
        )
        assert dataset_list[1].layout.prescan[1] == pytest.approx(
            dataset_1d_7.layout.prescan[1], 1.0e-4
        )

    clean(database_file=database_file)


def test__dataset_gen_from__analysis_use_dataset_full(
    dataset_1d_7, clocker_1d, samples_1d, model_1d
):
    dataset_1d_7_full = copy.copy(dataset_1d_7)
    dataset_1d_7_full.data[0] = 100.0

    analysis = ac.AnalysisDataset1D(
        dataset=dataset_1d_7, clocker=clocker_1d, dataset_full=dataset_1d_7_full
    )

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis + analysis,
        model=model_1d,
        samples=samples_1d,
    )

    dataset_agg = ac.agg.Dataset1DAgg(aggregator=agg, use_dataset_full=True)
    dataset_gen = dataset_agg.dataset_list_gen_from()

    for dataset_list in dataset_gen:
        assert dataset_list[0].data[0] == pytest.approx(100.0, 1.0e-4)
        assert dataset_list[1].data[0] == pytest.approx(100.0, 1.0e-4)
        assert dataset_list[0].layout.prescan[1] == pytest.approx(
            dataset_1d_7.layout.prescan[1], 1.0e-4
        )
        assert dataset_list[1].layout.prescan[1] == pytest.approx(
            dataset_1d_7.layout.prescan[1], 1.0e-4
        )

    clean(database_file=database_file)

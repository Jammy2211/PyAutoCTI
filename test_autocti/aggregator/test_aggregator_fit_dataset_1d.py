import pytest

import autocti as ac

from test_autocti.aggregator.conftest import clean, aggregator_from

database_file = "db_fit_dataset_1d"


def test__fit_dataset_1d_randomly_drawn_via_pdf_gen_from(
    dataset_1d_7, clocker_1d, samples_1d, model_1d
):
    analysis = ac.AnalysisDataset1D(dataset=dataset_1d_7, clocker=clocker_1d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model_1d,
        samples=samples_1d,
    )

    fit_dataset_1d_agg = ac.agg.FitDataset1DAgg(aggregator=agg)
    fit_dataset_1d_pdf_gen = fit_dataset_1d_agg.randomly_drawn_via_pdf_gen_from(
        total_samples=2
    )

    i = 0

    for fit_dataset_1d_gen in fit_dataset_1d_pdf_gen:
        for fit_dataset_1d_list in fit_dataset_1d_gen:
            i += 1

            assert fit_dataset_1d_list[0].post_cti_data[0] == pytest.approx(1.0, 1.0e-4)

    assert i == 2

    clean(database_file=database_file)


def test__fit_dataset_1d_all_above_weight_gen(
    dataset_1d_7, clocker_1d, samples_1d, model_1d
):
    analysis = ac.AnalysisDataset1D(dataset=dataset_1d_7, clocker=clocker_1d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model_1d,
        samples=samples_1d,
    )

    fit_dataset_1d_agg = ac.agg.FitDataset1DAgg(aggregator=agg)
    fit_dataset_1d_pdf_gen = fit_dataset_1d_agg.all_above_weight_gen_from(
        minimum_weight=-1.0
    )

    i = 0

    for fit_dataset_1d_gen in fit_dataset_1d_pdf_gen:
        for fit_dataset_1d_list in fit_dataset_1d_gen:
            i += 1

    assert i == 2

    clean(database_file=database_file)

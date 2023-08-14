import pytest

import autocti as ac

from test_autocti.aggregator.conftest import clean, aggregator_from

database_file = "db_cti_gen"


def test__cti_randomly_drawn_via_pdf_gen_from(
    dataset_1d_7, clocker_1d, samples_1d, model_1d
):
    analysis = ac.AnalysisDataset1D(dataset=dataset_1d_7, clocker=clocker_1d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model_1d,
        samples=samples_1d,
    )

    cti_agg = ac.agg.CTIAgg(aggregator=agg)
    cti_pdf_gen = cti_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for cti_gen in cti_pdf_gen:
        for cti in cti_gen:
            i += 1

            assert cti.trap_list[0].density == pytest.approx(0.1, 1.0e-4)

    assert i == 2

    clean(database_file=database_file)


def test__cti_all_above_weight_gen(dataset_1d_7, clocker_1d, samples_1d, model_1d):
    analysis = ac.AnalysisDataset1D(dataset=dataset_1d_7, clocker=clocker_1d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model_1d,
        samples=samples_1d,
    )

    cti_agg = ac.agg.CTIAgg(aggregator=agg)
    cti_pdf_gen = cti_agg.all_above_weight_gen_from(minimum_weight=-1.0)

    i = 0

    for cti_gen in cti_pdf_gen:
        for cti in cti_gen:
            i += 1

            if i == 1:
                assert cti.trap_list[0].density == pytest.approx(0.1, 1.0e-4)

            if i == 2:
                assert cti.trap_list[0].density == pytest.approx(0.1, 1.0e-4)

    assert i == 2

    clean(database_file=database_file)

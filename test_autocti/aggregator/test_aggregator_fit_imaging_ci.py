import autocti as ac

from test_autocti.aggregator.conftest import clean, aggregator_from

database_file = "db_fit_imaging_ci"


def test__fit_imaging_ci_randomly_drawn_via_pdf_gen_from(
    imaging_ci_7x7, parallel_clocker_2d, samples_2d, model_2d
):
    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model_2d,
        samples=samples_2d,
    )

    fit_agg = ac.agg.FitImagingCIAgg(aggregator=agg)
    fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1

            assert fit_list[0].post_cti_data[0] is not None

    assert i == 2

    clean(database_file=database_file)


def test__fit_imaging_ci_randomly_drawn_via_pdf_gen_from__multi_analysis(
    imaging_ci_7x7, parallel_clocker_2d, samples_2d, model_2d
):
    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis + analysis,
        model=model_2d,
        samples=samples_2d,
    )

    fit_agg = ac.agg.FitImagingCIAgg(aggregator=agg)
    fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1

            assert fit_list[0].post_cti_data[0] is not None
            assert fit_list[1].post_cti_data[0] is not None

    assert i == 2

    clean(database_file=database_file)


def test__fit_imaging_ci_all_above_weight_gen(
    imaging_ci_7x7, parallel_clocker_2d, samples_2d, model_2d
):
    analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model_2d,
        samples=samples_2d,
    )

    fit_agg = ac.agg.FitImagingCIAgg(aggregator=agg)
    fit_pdf_gen = fit_agg.all_above_weight_gen_from(minimum_weight=-1.0)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1

            assert fit_list[0].post_cti_data[0] is not None

    assert i == 2

    clean(database_file=database_file)

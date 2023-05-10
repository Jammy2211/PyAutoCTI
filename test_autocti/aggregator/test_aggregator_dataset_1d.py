from os import path

from autoconf import conf
import autofit as af
import autocti as ac

from test_autogalaxy.aggregator.conftest import clean


def test__dataset_generator_from_aggregator(dataset_1d_7, mask_1d_7_unmasked, clocker_1d, samples_1d, model_1d):

    path_prefix = "aggregator_dataset_gen"

    database_file = path.join(conf.instance.output_path, "dataset.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    masked_dataset_1d_7 = dataset_1d_7.apply_mask(mask=mask_1d_7_unmasked)

    search = ac.m.MockSearch(samples=samples_1d)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = ac.AnalysisDataset1D(dataset=masked_dataset_1d_7, clocker=clocker_1d)

    search.fit(model=model_1d, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    dataset_agg = ac.agg.Dataset1DAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_gen_from()

    for dataset in dataset_gen:

        assert (dataset.data == masked_dataset_1d_7.data).all()

    clean(database_file=database_file, result_path=result_path)

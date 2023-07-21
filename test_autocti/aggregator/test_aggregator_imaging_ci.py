from os import path

from autoconf import conf
import autofit as af
import autocti as ac

from test_autocti.aggregator.conftest import clean


# def test__dataset_gen_from(imaging_ci_7x7, parallel_clocker_2d, samples_2d, model_2d):
#     path_prefix = "aggregator_dataset_gen"
#
#     database_file = path.join(conf.instance.output_path, "dataset.sqlite")
#     result_path = path.join(conf.instance.output_path, path_prefix)
#
#     clean(database_file=database_file, result_path=result_path)
#
#     search = ac.m.MockSearch(samples=samples_2d)
#     search.paths = af.DirectoryPaths(path_prefix=path_prefix)
#
#     analysis = ac.AnalysisImagingCI(dataset=imaging_ci_7x7, clocker=parallel_clocker_2d)
#
#     search.fit(model=model_2d, analysis=analysis)
#
#     agg = af.Aggregator.from_database(filename=database_file)
#     agg.add_directory(directory=result_path)
#
#     dataset_agg = ac.agg.ImagingCIAgg(aggregator=agg)
#     dataset_gen = dataset_agg.dataset_list_gen_from()
#
#     for dataset_list in dataset_gen:
#         assert (dataset_list[0].data == imaging_ci_7x7.data).all()
#
#     clean(database_file=database_file, result_path=result_path)
#
#     analysis_list = [analysis, analysis]
#
#     analysis = sum(analysis_list)
#
#     search.fit(model=model_2d, analysis=analysis)
#
#     agg = af.Aggregator.from_database(filename=database_file)
#     agg.add_directory(directory=result_path)
#
#     dataset_agg = ac.agg.ImagingCIAgg(aggregator=agg)
#     dataset_gen = dataset_agg.dataset_list_gen_from()
#
#     for dataset_list in dataset_gen:
#         assert (dataset_list[0].data == imaging_ci_7x7.data).all()
#         assert (dataset_list[1].data == imaging_ci_7x7.data).all()

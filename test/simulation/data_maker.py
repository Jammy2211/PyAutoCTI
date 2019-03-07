

def simulate_integration_quadrant(test_name, normalizations, cti_params, cti_settings):
    output_path = "{}/data/".format(os.path.dirname(os.path.realpath(__file__))) + test_name + '/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sim_ci_patterns = ci_pattern.uniform_from_lists(normalizations=normalizations, regions=ci_regions)

    sim_ci_pre_ctis = list(map(lambda ci_pattern : ci_pattern.simulate_ci_pre_cti(shape=shape), sim_ci_patterns))

    sim_ci_datas = list(map(lambda ci_pre_cti, pattern:
                            ci_data.simulate(ci_pre_cti=ci_pre_cti, frame_geometry=frame_geometry,
                                             ci_pattern=pattern, cti_settings=cti_settings,
                                             cti_params=cti_params,
                                             read_noise=None),
                            sim_ci_pre_ctis, sim_ci_patterns))

    list(map(lambda sim_ci_data, index:
             util.numpy_array_to_fits(array=sim_ci_data.image, file_path=output_path + '/ci_image_' + str(index) + '.fits'),
             sim_ci_datas, range(len(sim_ci_datas))))

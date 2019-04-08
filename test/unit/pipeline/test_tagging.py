from autocti.pipeline import tagging

class TestPhaseTag:

    def test__mixture_of_values(self):

        phase_tag = tagging.phase_tag_from_phase_settings(columns=1, rows=(0,1),
                                                          cosmic_ray_parallel_buffer=1, cosmic_ray_serial_buffer=2,
                                                          cosmic_ray_diagonal_buffer=3)


        assert phase_tag == '_col_1_rows_(0,1)_cr_p1s2d3'

        phase_tag = tagging.phase_tag_from_phase_settings(columns=2, rows=(1,2),
                                                          cosmic_ray_parallel_buffer=4, cosmic_ray_serial_buffer=5,
                                                          cosmic_ray_diagonal_buffer=6)


        assert phase_tag == '_col_2_rows_(1,2)_cr_p4s5d6'


class TestTaggers:

    def test__columns_tagger(self):

        tag = tagging.columns_tag_from_columns(columns=None)
        assert tag == ''
        tag = tagging.columns_tag_from_columns(columns=10)
        assert tag == '_col_10'
        tag = tagging.columns_tag_from_columns(columns=60)
        assert tag == '_col_60'
        
    def test__rows_tagger(self):

        tag = tagging.rows_tag_from_rows(rows=None)
        assert tag == ''
        tag = tagging.rows_tag_from_rows(rows=(0, 5))
        assert tag == '_rows_(0,5)'
        tag = tagging.rows_tag_from_rows(rows=(10, 20))
        assert tag == '_rows_(10,20)'

    def test__cosmic_ray_buffer_tagger(self):

        tag = tagging.cosmic_ray_buffer_tag_from_cosmic_ray_buffers(cosmic_ray_parallel_buffer=1,
                                                                    cosmic_ray_serial_buffer=2,
                                                                    cosmic_ray_diagonal_buffer=3)
        assert tag == '_cr_p1s2d3'

        tag = tagging.cosmic_ray_buffer_tag_from_cosmic_ray_buffers(cosmic_ray_parallel_buffer=10,
                                                                    cosmic_ray_serial_buffer=5,
                                                                    cosmic_ray_diagonal_buffer=1)
        assert tag == '_cr_p10s5d1'
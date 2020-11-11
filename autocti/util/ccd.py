from arcticpy import ccd

class CCD(ccd.CCD):
    def __init__(self, full_well_depth=1e4, well_notch_depth=0.0, well_fill_power=0.58):
        super().__init__(
            fraction_of_traps_per_phase=[1],
            full_well_depth=full_well_depth,
            well_fill_power=well_fill_power,
            well_notch_depth=well_notch_depth,
            well_bloom_level=None,
        )

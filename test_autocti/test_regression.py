model = af.Collection(
    cti=af.Model(ac.CTI2D, parallel_trap_list=traps_x1, parallel_ccd=ccd),
    hyper_noise=af.Model(ac.HyperCINoiseCollection),
)

print(model.has(ac.HyperCINoiseCollection))

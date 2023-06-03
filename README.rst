PyAutoCTI: Charge Transfer Inefficiency Modeling
================================================

.. |nbsp| unicode:: 0xA0
    :trim:

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/Jammy2211/autocti_workspace/HEAD

.. |RTD| image:: https://readthedocs.org/projects/pyautocti/badge/?version=latest
    :target: https://pyautocti.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |Tests| image:: https://github.com/Jammy2211/PyAutoCTI/actions/workflows/main.yml/badge.svg
   :target: https://github.com/Jammy2211/PyAutoCTI/actions

.. |Build| image:: https://github.com/Jammy2211/PyAutoBuild/actions/workflows/release.yml/badge.svg
   :target: https://github.com/Jammy2211/PyAutoBuild/actions

.. |code-style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |arXiv| image:: https://img.shields.io/badge/arXiv-1708.07377-blue
    :target: https://arxiv.org/abs/0909.0507

|binder| |RTD| |Tests| |Build| |code-style| |arXiv|

`Installation Guide <https://pyautocti.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautocti.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Binder <https://mybinder.org/v2/gh/Jammy2211/autocti_workspace/release?filepath=introduction.ipynb>`_ |
`What is CTI? <https://pyautocti.readthedocs.io/en/latest/overview/overview_1_what_is_cti.html>`_

Charge Transfer Inefficiency, or CTI for short, is an effect that occurs when acquiring imaging data from
Charge Coupled Devices (CCDs). Due to radiation damage to the CCD's silicon lattice electrons are read-out inefficiently,
creating a characteristic trailing or smearing effect.

Here is an example of CTI in the Hubble space telescope, after decades of radiation damage:

.. image:: https://raw.githubusercontent.com/Jammy2211/PyAutoCTI/main/docs/overview/images/what_is_cti.png
  :width: 600
  :alt: Alternative text

**PyAutoCTI** makes it simple to calibrate a time-varying CTI model using in-orbit observations and correct CTI in
science imaging using this model.

**PyAutoCTI** development is centred around mitigating CTI for the Euclid space mission, which relies on the precise
measurement of galaxy shapes in order to map out the distribution of dark matter throughout the Universe via a
phenomena called weak gravitational lensing.

Getting Started
---------------

The following links are useful for new starters:

- `The introduction Jupyter Notebook on Binder <https://mybinder.org/v2/gh/Jammy2211/autocti_workspace/release?filepath=introduction.ipynb>`_, where you can try **PyAutoCTI** in a web browser (without installation).

- `The PyAutoCTI readthedocs <https://pyautocti.readthedocs.io/en/latest>`_, which includes `an installation guide <https://pyautocti.readthedocs.io/en/latest/installation/overview.html>`_ and an overview of **PyAutoCTI**'s core features.

- `The autocti_workspace GitHub repository <https://github.com/Jammy2211/autocti_workspace>`_, which includes example scripts and the `HowToCTI Jupyter notebook tutorials <https://github.com/Jammy2211/autocti_workspace/tree/master/notebooks/howtocti>`_ which give new users a step-by-step introduction to **PyAutoCTI**.

API Overview
------------

To model CTI, **PyAutoCTI** wraps the library **arCTIc** (https://github.com/jkeger/arctic).

CTI can be added to an image as follows:

.. code-block:: python

    import autocti as ac

    """
    Define a pre-cti image which **PyAutoCTI** adds CTI to.
    """
    pre_cti_data_2d = ac.Array2D.no_mask(
                values=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        pixel_scales=0.1,
    )

    """
    A clocker object is used to model CCD clocking, which includes customization such
    as the properties of the read-out electronics.
    """
    clocker_2d = ac.Clocker2D(parallel_roe=ac.ROE())

    """
    CTI is caused by traps on the CCD's silicon lattice, for example traps which capture
    electrons instantaneously.
    """
    parallel_trap = ac.TrapInstantCapture(density=100.0, release_timescale=1.0)

    """
    CTI also depends on how electrons fill each pixel in a CCD, therefore we define
    the volume-filling properties of the CCD.
    """
    parallel_ccd = ac.CCDPhase(
        well_fill_power=0.58, well_notch_depth=0.0, full_well_depth=200000.0
    )

    """
    The data, traps and CCD properties are combined to clock the pre-CTI data and return the
    post-CTI data.
    """
    post_cti_data_2d = clocker_2d.add_cti(
        data=pre_cti_data_2d,
        parallel_trap_list=[parallel_trap],
        parallel_ccd=parallel_ccd
    )

    """
    We can use PyAutoCTI's built in visualization library to plot the data with CTI.
    """
    import autocti.plot as aplt

    array_2d_plotter = aplt.Array2DPlotterarray=post_cti_data_2d)
    array_2d_plotter.figure_2d()


With **PyAutoCTI**, you can begin calibrating a CTI model in minutes. The example below demonstrates a simple analysis
which fits a CTI model to charge injection imaging calibrate data (a form of data used to calibrate a CTI model)

.. code-block:: python

    import autofit as af
    import autocti as al
    import autocti.plot as aplt

    """
    Define the 2D shape of the charge injection image.
    """
    shape_native = (30, 30)

    """
    Define where the charge injection is on the data.
    """
    regions_list = [(0, 25, serial_prescan[3], serial_overscan[2])]

    """
    Setup the data layout which informs **PyAutoCTI** where information on 
    CTI is in the data.
    """
    layout = ac.Layout2DCI(
        shape_2d=shape_native,
        region_list=regions_list,
    )

    """
    Load the charge injection image from fits.
    """
    dataset = ac.ImagingCI.from_fits(
        data_path=path.join(dataset_path, f"data.fits"),
        noise_map_path=path.join(dataset_path, f"noise_map.fits"),
        pre_cti_data_path=path.join(dataset_path, f"pre_cti_data.fits"),
        layout=layout,
        pixel_scales=0.1,
    )

    """
    Again define the clocker which models CCD clocking and read-out electronics.
    """
    clocker_2d = ac.Clocker2D(parallel_roe=ac.ROE())

    """
    Define the traps in the CTI model and customize the priors of their free parameters.
    """
    trap = af.Model(ac.TrapInstantCapture)
    
    trap.density = af.UniformPrior(lower_limit=0.0, upper_limit=20.0)
    trap.release_timescale = af.UniformPrior(lower_limit=0.0, upper_limit=20.0)

    """
    Define the CCD filling behaviour of the CTI, which is also part of the model and is
    fitted for as free parameters.
    """
    parallel_ccd = af.Model(ac.CCDPhase)

    parallel_ccd.well_fill_power = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    parallel_ccd.well_notch_depth = 0.0
    parallel_ccd.full_well_depth = 200000.0

    """
    We define the non-linear search used to fit the model to the data (in this case, Dynesty).
    """
    search = af.DynestyStatic(name="search[example]", nlive=50)

    """
    We next set up the `Analysis`, which contains the `log likelihood function` that the
    non-linear search calls to fit the cti model to the data.
    """
    analysis = ac.AnalysisImagingCI(dataset=dataset, clocker=clocker_2d)

    """
    To perform the model-fit we pass the model and analysis to the search's fit method. This will
    output results (e.g., dynesty samples, model parameters, visualization) to hard-disk.
    """
    result = search.fit(model=model, analysis=analysis)

    """
    The results contain information on the fit, for example the maximum likelihood
    model from the Dynesty parameter space search.
    """
    print(result.samples.max_log_likelihood())

Support
-------

Support for installation issues, help with cti modeling and using **PyAutoCTI** is available by
`raising an issue on the GitHub issues page <https://github.com/Jammy2211/PyAutoCTI/issues>`_.

We also offer support on the **PyAutoCTI** `Slack channel <https://pyautocti.slack.com/>`_, where we also provide the
latest updates on **PyAutoCTI**. Slack is invitation-only, so if you'd like to join send
an `email <https://github.com/Jammy2211>`_ requesting an invite.

.. _database:

Database
--------

The default behaviour of **PyAutoCTI** is for model-fitting results to be output to hard-disc in folders, which are
straight forward to navigate and manually check the CTI modeling results. For small samples of CTI calibration datasets
this is sufficient, however many users have a need to perform many model fits to large CTI datasets, making the manual
inspection of results time consuming.

This is especially true for the CTI monitoring of space satellites like Euclid of the Hubble Space Telescope, which
acquire CTI calibration data every day in order to update the CTI model used to correct science imaging daily. The
**PyAutoCTI** database features are built to facilitate this in-orbit calibration and allow for the global CTI
model to be updated using years worth of data in a fully Bayesian way.

**PyAutoCTI**'s database feature outputs all model-fitting results as a sqlite3 (https://docs.python.org/3/library/sqlite3.html)
relational database, such that all results can be efficiently loaded into a Jupyter notebook or Python script for
inspection, analysis and interpretation. This database supports advanced querying, so that specific
model-fits (e.g., which fit a certain CTI model or dataset) can be loaded.

To make it so that results are output to an .sqlite database we simply open a database session and pass this session
to the non-linear search:

.. code-block:: python

    session = af.db.open_database("database.sqlite")

    emcee = af.Emcee(
        session=session,  # This instructs the search to write to the .sqlite database.
    )

When a model-fit is performed, a unique identifier is generated based on the model and non-linear search. However,
if we were to fit many different CTI datasets with the same model and non-linear search, they would all use the same
unique identifier and not be distinguishable by the database.

We can overcome this by using the name of the dataset as the ``unique_tag`` passed to the search, which is used
alongside the model and search to create the unique identifier:

.. code-block:: python

    session = af.db.open_database("database.sqlite")

    dataset_name = "CTI_dataset_0"

    emcee = af.Emcee(
        unique_tag=dataset_name,  # This makes the unique identifier use the dataset name
        session=session,  # This instructs the search to write to the .sqlite database.
    )

Lets suppose that we have performed 100 model-fits to 100 CTI datasets, and when we ran **PyAutoCTI** we told it
to write to the ``.sqlite`` database file. We can load these results in a Python script or Jupyter notebook using
the ``Aggregator``:

.. code-block:: python

    agg = Aggregator.from_database("path/to/output/database.sqlite")

We can now use the ``Aggregator`` to inspect the results of all model-fits. For example, we can load the ``Samples``
object of all 100 model-fits, which contains information on the best-fit model, posterior, Bayesian evidence, etc.

Below, we use the samples generator to create a list of the maximum log likelihood of every model-fit and print it:

.. code-block:: python

    for samples in agg.values("samples"):

        print(max(samples.log_likelihood))

This object (and all objects loaded by the ``Aggregator``) are returned as a generator (as opposed to a list,
dictionary or other Python type). This is because generators do not store large arrays or classes in memory until they
are used, ensuring that when we are manipulating large sets of results we do not run out of memory!

We can iterate over the samples to print the maximum log likelihood CTI model of every fit:

.. code-block:: python

    for samps in agg.values("samples"):

        instance = samps.max_log_likelihood()

        print("Maximum Likelihood Model-fit \n")
        print(instance)


The ``Aggregator`` contains tools for querying the database for certain results, for example to load subsets of
model-fits. This can be done in many different ways, depending on what information you want.

Below, we query based on the model fitted. For example, we can load all results which fitted an ``TrapInstantCapture``
parallel CTI model (note that when we performed the model fit, we composed the CTI model using the string ``cti``
which has a parallel trap list named ``parallel_traps``):

.. code-block:: python

    cti = agg.model.cti
    agg_query = agg.query(cti.parallel_traps[0] == ac.TrapInstantCapture)

    samples_gen = agg_query.values("samples")

Queries using the results of model-fitting are also supported. Below, we query the database to find all fits where the
inferred value of the ``density`` for the ``TrapInstantCapture`` is above 1.0:

.. code-block:: python

    cti = agg.model.cti
    agg_query = agg.query(cti.parallel_traps[0].density > 1.0)

Advanced queries can be constructed using logic, for example we below we combine the two queries above to find all
results which fitted an ``TrapInstantCapture`` AND (using the & symbol) inferred a value of density above 1.0.

The OR logical clause is also supported via the symbol |.

.. code-block:: python

    cti = agg.model.cti
    agg_query = agg.query((cti.parallel_traps[0] == ac.TrapInstantCapture) & (cti.parallel_traps[0].density > 1.0))

We can query using the ``unique_tag`` to load the model-fit to a specific dataset:

.. code-block:: python

    agg_query = agg.query(agg.unique_tag == "example_dataset_0")

An ``info`` dictionary can be passed into a model-fit, which contains information on the model-fit. The example below
creates an ``info`` dictionary which is passed to the model-fit, which is then loaded via the database.

.. code-block:: python

    info = {"example_key": "example_value"}

    emcee.fit(model=model, analysis=analysis, info=info)

    agg = Aggregator.from_database("path/to/output/database.sqlite")

    info_gen = agg.values("info")

Databases are an extremely powerful feature for users tasked with fitting extremely large datasets as well as fitting
many different CTI models, where the scale of the problem can make the management of the large quantity of results
produced prohibitive. This is especially true on high performance computing facilities, which often have restrictions
on the number of files that a user can store on the machine.

Furthermore, if you are using the search chaining or graphical modeling features you'll be aware that one can easily
start generating thousands or *millions* of model-fits. There is no way to manage this large library of results other
than the database!

A complete description of the database is given in
the `database folder on the autocti_workspace <https://github.com/Jammy2211/autoCTI_workspace/tree/release/notebooks/database>`_.
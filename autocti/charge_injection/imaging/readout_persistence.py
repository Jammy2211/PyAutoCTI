import numpy as np
from typing import Optional, Tuple

import autoarray as aa


class ReadoutPersistence:
    def __init__(
        self,
        total_rows: Optional[int] = 10,
        mean: Optional[float] = 5.0,
        sigma: Optional[float] = 1.0,
        rows_per_persistence_range: Optional[Tuple[int, int]] = (1, 10),
        seed: int = -1,
    ):
        """
        Adds readout persistence to a 2D array of data, typically for simulating charge injection data.

        Readout persistence is a feature of CCD detectors (e.g. the Euclid VIS detector), whereby after bright
        delta-function like sources (e.g. cosmic rays) are detected, the electronics does not properly reset its bias
        level to zero, but instead gets stuck at a higher value (e.g. 3e- to 10e-). Tis persistence appears as rows of
        elevated signal in the charge injection data.

        If not masked, this elevated signal appears as spikes in the parallel EPER trails, degrading the quality of
        CTI calibration.

        This class adds readout persistence to a 2D array of data, whereby a number of rows are selected at random and
        a value drawn from a Gaussian distribution is added to them. The number of rows and value drawn from the
        Gaussian distribution are input as parameters.

        This simulation does not pair the location of readout persistence with the location of the location of delta
        function like sources, but instead adds it to random rows in the data. This choice is for simplicity and
        because all masking techniques do not use the point-source locations to mask readout persistence.

        Parameters
        ----------
        total_rows
            The total number of distinct rows where readout persistence is added to the data. Note that for each row
            multiple rows are added, drawn randomly from the range `rows_per_persistence_range`.
        mean
            The mean of the Gaussian distribution from which the value added to the data is drawn.
        sigma
            The sigma of the Gaussian distribution from which the value added to the data is drawn.
        rows_per_persistence_range
            The range of the number of rows added to the data for each row where readout persistence is added.
        seed
            The seed used to initialize the random number generator. If -1, a random seed is used.
        """

        self.total_rows = total_rows
        self.mean = mean
        self.sigma = sigma
        self.rows_per_persistence_range = rows_per_persistence_range
        self.seed = seed

    def data_with_readout_persistence_from(self, data: aa.Array2D) -> aa.Array2D:
        """
        Returns the input data with readout persistence added to it.

        This function adds readout persistence to the input data, whereby a number of rows are selected at random and
        a value drawn from a Gaussian distribution is added to them. The number of rows and value drawn from the
        Gaussian distribution are input as parameters.

        The `__init__` method describes how the readout persistence is simulated.

        Parameters
        ----------
        data
            The 2D array of data to which readout persistence is added.

        Returns
        -------
        The input data with readout persistence added to it.
        """
        for i in range(self.total_rows):
            if self.seed == -1:
                seed = np.random.randint(0, int(1e9))
            else:
                seed = self.seed + i

            np.random.seed(seed)

            row_value = 0.0

            while row_value <= 0.0:
                row_value = np.random.normal(self.mean, self.sigma)

            row_index = np.random.randint(0, data.shape[0])
            row_range = np.random.randint(
                self.rows_per_persistence_range[0], self.rows_per_persistence_range[1]
            )

            data[row_index : row_index + row_range] += row_value

        return data

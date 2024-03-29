import numpy as np
from typing import Optional, Tuple


class ReadoutPersistence:

    def __init__(
        self,
        total_rows: Optional[int] = 10,
        mean : Optional[float] = 5.0,
        sigma: Optional[float] = 1.0,
        rows_per_persistence_range: Optional[Tuple[int, int]] =  (1, 10),
    ):

        self.total_rows = total_rows
        self.mean = mean
        self.sigma = sigma
        self.rows_per_persistence_range = rows_per_persistence_range

    def data_with_readout_persistence_from(self, data):

        for i in range(self.total_rows):

            row_value = 0.0

            while row_value <= 0.0:
                row_value = np.random.normal(self.mean, self.sigma)

            row_index = np.random.randint(0, data.shape[0])
            row_range = np.random.randint(1, 10)

            data[row_index:row_range] += row_value

        return data
import numpy as np
from typing import Optional, Tuple


class ReadoutPersistence:

    def __init__(
        self,
        total_rows: Optional[int] = 10,
        mean : Optional[float] = 5.0,
        sigma: Optional[float] = 1.0,
        rows_per_persistence_range: Optional[Tuple[int, int]] =  (1, 10),
        seed: int = -1,
    ):

        self.total_rows = total_rows
        self.mean = mean
        self.sigma = sigma
        self.rows_per_persistence_range = rows_per_persistence_range
        self.seed = seed

    def data_with_readout_persistence_from(self, data):

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
            row_range = np.random.randint(self.rows_per_persistence_range[0], self.rows_per_persistence_range[1])

            print(row_value)
            print(row_index)
            print(row_range)

            data[row_index:row_index+row_range] += row_value

        return data
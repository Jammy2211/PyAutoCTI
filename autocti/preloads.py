import numpy as np
from typing import Optional


class Preloads:
    def __init__(
        self,
        parallel_fast_index_list: Optional[np.ndarray] = None,
        parallel_fast_column_lists: Optional[np.ndarray] = None,
    ):

        self.parallel_fast_index_list = parallel_fast_index_list
        self.parallel_fast_column_lists = parallel_fast_column_lists

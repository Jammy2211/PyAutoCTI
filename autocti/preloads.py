import numpy as np
from typing import Optional


class Preloads:
    def __init__(
        self,
        parallel_fast_index_list: Optional[np.ndarray] = None,
        parallel_fast_column_lists: Optional[np.ndarray] = None,
        serial_fast_index_list: Optional[np.ndarray] = None,
        serial_fast_row_lists: Optional[np.ndarray] = None,
        noise_normalization: Optional[float] = None,
    ):
        """
        Class which offers a concise API for settings up the preloads, which before a model-fit are set up via
        inspection of attributes such as the clocker.

        For example, if the clocker's `parallel_fast_mode` is on, this significantly reduces the number of arCTIc
        calls in CTI calibration data. The method inspects every image passed to arctic, extracts
        all unique columns, only passes these to arCTIc and use the output to rebuild the `post_cti_data`. The indexes
        which store unique columns and map them to the post-CTI data can be preloaded in memory to avoid repeated
        calculations in the likelihood function.


        Parameters
        ----------
        parallel_fast_index_list
            The index of a column that is repeated in the pre-cti data. This index corresponds to the first index
            of the repeated columns and this array used to extract the columns from the pre-cti data which are passed
            to arctic.
        parallel_fast_column_lists
            The mapping of every repeated column in `parallel_fast_index_list`  to all other columns which are identical.
             This is used to map the reduced arCTIc output to the post-CTI data.
        serial_fast_index_list
            The index of a row that is repeated in the pre-cti data. This index corresponds to the first index
            of the repeated rows and this array used to extract the rows from the pre-cti data which are passed
            to arctic.
        serial_fast_row_lists
            The mapping of every repeated row in `serial_fast_index_list`  to all other rows which are identical.
            This is used to map the reduced arCTIc output to the post-CTI data.
        noise_normalization
            The noise normalization term of the log likelihood function evaluated in `Analysis` objects. If the
            noise-map is fixed, this can be preloaded as it does not change.

        Returns
        -------
        Preloads
            The preloads object used to skip certain calculations in the log likelihood function.
        """
        self.parallel_fast_index_list = parallel_fast_index_list
        self.parallel_fast_column_lists = parallel_fast_column_lists
        self.serial_fast_index_list = serial_fast_index_list
        self.serial_fast_row_lists = serial_fast_row_lists
        self.noise_normalization = noise_normalization

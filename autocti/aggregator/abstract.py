from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from autocti.clocker.abstract import AbstractClocker

import autofit as af


class AbstractAgg(af.AbstractAgg):
    def __init__(
        self,
        aggregator: af.Aggregator,
        use_dataset_full: bool = False,
        clocker_list: Optional[List[AbstractClocker]] = None,
    ):
        """
        An abstract aggregator wrapper, which makes it straight forward to compute generators of objects from specific
        samples of a non-linear search.

        For example, in **PyAutoLens**, this makes it straight forward to create generators of `Plane`'s drawn from
        the PDF estimated by a non-linear for efficient error calculation of derived quantities.
        Parameters
        ----------
        aggregator
            An PyAutoFit aggregator containing the results of non-linear searches performed by PyAutoFit.
        clocker
            The CTI arctic clocker used by aggregator's instances. If None is input, the clocker used by the
            non-linear search and model-fit is used.
        """
        super().__init__(aggregator=aggregator)

        self.use_dataset_full = use_dataset_full
        self.clocker_list = clocker_list

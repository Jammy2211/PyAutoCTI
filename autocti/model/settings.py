from typing import List, Optional, Tuple

from arcticpy import TrapInstantCapture

from autoconf.dictable import from_dict, from_json, output_to_json

from autocti import exc


class AbstractSettingsCTI:
    def check_total_density_within_range_of_traps(
        self, total_density_range: Tuple[float, float], traps: List[TrapInstantCapture]
    ):
        if total_density_range is not None:
            total_density = sum([trap.density for trap in traps])

            if (
                total_density < total_density_range[0]
                or total_density > total_density_range[1]
            ):
                raise exc.PriorException

    @classmethod
    def from_json(cls, file_path):
        return from_json(file_path=file_path)

    def output_to_json(self, file_path):
        output_to_json(obj=self, file_path=file_path)

class SettingsCTI1D(AbstractSettingsCTI):
    def __init__(
        self,
        total_density_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Controls the modeling settings of CTI clocking in 1D.

        For a CTI model-fit in 1D, the total density of all traps can be constrained to be within an input range of
        values.

        For example, if `total_density_range=(0.0, 0.5)`, the total density of all traps will be between 0.0 and 0.5
        electrons per pixel. This acts on the total density of traps, and therefore sums up the individual density
        of each trap species.

        Parameters
        ----------
        total_density_range
            The range of total trap density values allowed by the model-fit.
        """
        self.total_density_range = total_density_range

    def check_total_density_within_range(self, traps: List[TrapInstantCapture]):
        """
        Checks that the total density of traps is within the input range.

        This function receives as input lists of trap species and checks that their total density is within the
        range specified by the `total_density_range` attribute.

        If they are not in this range, an exception is raised, meaning that the model is resampled and this model is
        rejected.

        Parameters
        ----------
        traps
            The list of trap species whose densities are checked.
        """
        self.check_total_density_within_range_of_traps(
            total_density_range=self.total_density_range, traps=traps
        )


class SettingsCTI2D(AbstractSettingsCTI):
    def __init__(
        self,
        parallel_total_density_range: Optional[Tuple[float, float]] = None,
        serial_total_density_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Controls the modeling settings of CTI clocking in 2D.

        For a CTI model-fit in 2D, the total density of all parallel and / or serial traps can be constrained to be
        within an input range of  values.

        For example, if `parallel_total_density_range=(0.0, 0.5)`, the total density of all parallel traps will be
        between 0.0 and 0.5  electrons per pixel. This acts on the total density of traps, and therefore sums up
        the individual density of each parallel trap species.

        Parameters
        ----------
        parallel_total_density_range
            The range of parallel total trap density values allowed by the model-fit.
        serial_total_density_range
            The range of serial total trap density values allowed by the model-fit.
        """
        self.parallel_total_density_range = parallel_total_density_range
        self.serial_total_density_range = serial_total_density_range

    def check_total_density_within_range(self, parallel_traps, serial_traps):
        """
        Checks that the total density of traps is within the input range.

        This function receives as input lists of parallel and / or serial trap species and checks that their total
        density is within the range specified by the `parallel_total_density_range` and `serial_total_density_range`
        attributes.

        If they are not in this range, an exception is raised, meaning that the model is resampled and this model is
        rejected.

        Parameters
        ----------
        parallel_traps
            The list of parallel trap species whose densities are checked.
        serial_traps
            The list of serial trap species whose densities are checked.
        """
        self.check_total_density_within_range_of_traps(
            total_density_range=self.parallel_total_density_range, traps=parallel_traps
        )

        self.check_total_density_within_range_of_traps(
            total_density_range=self.serial_total_density_range, traps=serial_traps
        )

from typing import Dict, Tuple, Optional

from autoarray.structures.header import Header


class HeaderEuclid(Header):
    def __init__(
        self,
        header_sci_obj: Dict = None,
        header_hdu_obj: Dict = None,
        original_roe_corner: Tuple[int, int] = None,
        readout_offsets: Optional[Tuple] = None,
        ccd_id: Optional[str] = None,
        quadrant_id: Optional[str] = None,
    ):
        super().__init__(
            header_sci_obj=header_sci_obj,
            header_hdu_obj=header_hdu_obj,
            original_roe_corner=original_roe_corner,
            readout_offsets=readout_offsets,
        )

        self.ccd_id = ccd_id
        self.quadrant_id = quadrant_id

    @property
    def row_index(self) -> str:
        if self.ccd_id is not None:
            return self.ccd_id[-1]

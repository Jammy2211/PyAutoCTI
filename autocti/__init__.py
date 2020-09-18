from arcticpy.traps import TrapInstantCaptureWrap as TrapInstantCapture
from arcticpy.traps import TrapInstantCaptureWrap
from arcticpy.main import ClockerWrap as Clocker
from arcticpy.main import ClockerWrap
from arcticpy.ccd import CCDWrap as CCD
from arcticpy.ccd import CCDWrap

from autoarray.fit.fit import FitImaging
from autoarray.structures.region import Region
from autoarray.structures.arrays.abstract_array import ExposureInfo
from autoarray.structures.frames.abstract_frame import Scans
from autoarray.instruments import euclid
from autoarray.instruments import acs
from autoarray.dataset import preprocess

from .structures.arrays import Array
from .structures.frame import Frame
from .mask.mask import Mask
from .mask.mask import SettingsMask
from .dataset.imaging import Imaging
from .dataset.imaging import MaskedImaging
from . import charge_injection as ci
from .pipeline.phase.settings import SettingsCTI
from .pipeline.phase.settings import SettingsPhaseCIImaging
from .pipeline.phase.extensions import CombinedHyperPhase
from .pipeline.phase.ci_imaging.phase import PhaseCIImaging
from .pipeline.pipeline import Pipeline
from . import util
from . import plot

__version__ = "0.11.3"

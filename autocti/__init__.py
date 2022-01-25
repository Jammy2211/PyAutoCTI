from autoarray.mask.mask_1d import Mask1D
from autoarray.layout.region import Region1D
from autoarray.layout.region import Region2D
from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.arrays.abstract_array import Header
from autoarray.instruments import euclid
from autoarray.instruments import acs
from autoarray.dataset import preprocess
from autoarray.dataset.imaging import Imaging

from arcticpy.src.roe import ROE
from arcticpy.src.roe import ROEChargeInjection
from arcticpy.src.ccd import CCDPhase
from arcticpy.src.ccd import CCD
from arcticpy.src.traps import TrapInstantCapture
from arcticpy.src.traps import TrapSlowCapture
from arcticpy.src.traps import TrapInstantCaptureContinuum

from .cosmics.cosmics import SimulatorCosmicRayMap
from .charge_injection.layout import Extractor2DParallelFrontEdge
from .charge_injection.layout import Extractor2DParallelTrails
from .charge_injection.layout import Extractor2DSerialFrontEdge
from .charge_injection.layout import Extractor2DSerialTrails
from .charge_injection.fit import FitImagingCI
from .mask.mask_2d import Mask2D
from .mask.mask_2d import SettingsMask2D
from .line.mask_1d import Mask1DLine
from .line.mask_1d import SettingsMask1DLine
from .line.layout import Extractor1DFrontEdge
from .line.layout import Extractor1DTrails
from .line.layout import Layout1DLine
from .line.dataset import SettingsDatasetLine
from .line.dataset import DatasetLine
from .line.dataset import SimulatorDatasetLine
from .line.fit import FitDatasetLine
from .line.model.analysis import AnalysisDatasetLine
from . import charge_injection as ci
from .charge_injection.model.analysis import AnalysisImagingCI
from .model.model_util import CTI1D
from .model.model_util import CTI2D
from .model.settings import SettingsCTI1D
from .model.settings import SettingsCTI2D
from .util.clocker import Clocker1D
from .util.clocker import Clocker2D
from . import util
from . import plot

from autoconf import conf

conf.instance.register(__file__)

__version__ = "2021.10.14.1"

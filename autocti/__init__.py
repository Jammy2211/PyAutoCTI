from autoarray.mask.mask_1d import Mask1D
from autoarray.layout.region import Region1D
from autoarray.layout.region import Region2D
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.header import Header
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

from .charge_injection.fit import FitImagingCI
from .charge_injection.hyper import HyperCINoiseScalar
from .charge_injection.hyper import HyperCINoiseCollection
from .charge_injection.imaging.imaging import ImagingCI
from .charge_injection.imaging.settings import SettingsImagingCI
from .charge_injection.imaging.simulator import SimulatorImagingCI
from .charge_injection.layout import Layout2DCI
from .cosmics.cosmics import SimulatorCosmicRayMap
from .extract.two_d.parallel_fpr import Extract2DParallelFPR
from .extract.two_d.parallel_eper import Extract2DParallelEPER
from .extract.two_d.serial_fpr import Extract2DSerialFPR
from .extract.two_d.serial_eper import Extract2DSerialEPER
from .extract.two_d.parallel_calibration import Extract2DParallelCalibration
from .extract.two_d.serial_calibration import Extract2DSerialCalibration
from .extract.two_d.master import Extract2DMaster
from .charge_injection.fit import FitImagingCI
from .mask.mask_2d import Mask2D
from .mask.mask_2d import SettingsMask2D
from .mask.mask_1d import Mask1D
from .mask.mask_1d import SettingsMask1D
from .extract.one_d.fpr import Extract1DFPR
from .extract.one_d.eper import Extract1DEPER
from .extract.one_d.master import Extract1DMaster
from .layout.one_d import Layout1D
from .dataset_1d.dataset_1d.settings import SettingsDataset1D
from .dataset_1d.dataset_1d.dataset_1d import Dataset1D
from .dataset_1d.dataset_1d.simulator import SimulatorDataset1D
from .dataset_1d.fit import FitDataset1D
from .dataset_1d.model.analysis import AnalysisDataset1D
from . import charge_injection as ci
from .charge_injection.model.analysis import AnalysisImagingCI
from .model.model_util import CTI1D
from .model.model_util import CTI2D
from .model.settings import SettingsCTI1D
from .model.settings import SettingsCTI2D
from .clocker.one_d import Clocker1D
from .clocker.two_d import Clocker2D
from . import util
from . import plot

from autoconf import conf

conf.instance.register(__file__)

__version__ = "2022.03.18.2"

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseConceptualModel(ABC):
    """Base class for conceptual models."""
    @staticmethod
    def precipitation(
        temp: float,
        ltt: float,
        utt: float,
        rfcf: float,
        sfcf: float,
        pcorr: float = 1.0,
    ) -> Tuple[float, float]:
        """Split precipitation into rainfall and snowfall."""
        ...

    @staticmethod
    def snow(
        temp, ttm, cfmax, cfr, cwh, rf, sf, wc_old, sp_old
    ) -> Tuple[float, float, float]:
        """Snow accumulation/melt routine."""
        ...

    @staticmethod
    def soil(
        fc,
        beta,
        etf,
        temp,
        tm,
        e_corr,
        lp,
        tfac,
        c_flux,
        infiltration,
        ep,
        sm_old,
        uz_old,
    ) -> Tuple[float, float]:
        """Soil moisture and runoff routine."""
        ...

    @staticmethod
    def response(
        tfac, perc, alpha, k, k1, area, lz_old, uz_int_1
    ) -> Tuple[float, float, float]:
        """Convert runoff to stream discharge."""
        ...

    def routing(self, q: np.ndarray, maxbas: int = 1) -> np.ndarray:
        """Apply triangular routing function."""
        ...

    @abstractmethod
    def simulate(
        self, prec, temp, et, ll_temp, par, init_st=None, q_init=None, snow=0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...

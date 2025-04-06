from typing import Tuple
import numpy as np

class BaseConceptualModel:
    def precipitation(
            self, temp: float, ltt: float, utt: float, prec: float, rfcf: float, sfcf: float, pcorr: float = 1.0
    ) -> Tuple[float, float]:
        """Split precipitation into rainfall and snowfall."""
        ...

    def snow(
            self, temp, ttm, cfmax, cfr, cwh, rf, sf, wc_old, sp_old
    ) -> Tuple[float, float, float]:
        """Snow accumulation/melt routine."""
        ...

    def soil(
            self, fc, beta, etf, temp, tm, e_corr, lp, tfac, c_flux,
            infiltration, ep, sm_old, uz_old
    ) -> Tuple[float, float]:
        """Soil moisture and runoff routine."""
        ...

    def response(
            self, tfac, perc, alpha, k, k1, area, lz_old, uz_int_1
    ) -> Tuple[float, float, float]:
        """Convert runoff to stream discharge."""
        ...

    def routing(
            self, q: np.ndarray, maxbas: int = 1
    ) -> np.ndarray:
        """Apply triangular routing function."""
        ...


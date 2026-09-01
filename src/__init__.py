"""PINN-Schrodinger: PINNs vs finite differences for the time-independent Schrodinger equation."""

import numpy as _np

# numpy<2.0 only has trapz; numpy>=2.0 renamed it to trapezoid. Alias so the rest of the
# package (and the notebooks) can call np.trapezoid regardless of the installed version.
if not hasattr(_np, "trapezoid"):
    _np.trapezoid = _np.trapz

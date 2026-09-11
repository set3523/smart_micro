"""Physical constants and shared configuration."""
import numpy as np

# Electromagnetic constants (microwave oven, ~2.45 GHz scaled)
freq = 2.45e9
omega = 2 * np.pi * freq
c0 = 299792458.0
mu0 = 4 * np.pi * 1e-7
eps0 = 1 / (mu0 * c0 ** 2)
k0 = omega / c0

# Chamber dimensions (m)
CHAMBER_SIZE = (0.31, 0.26, 0.31)

# Material properties (pork / food)
EPS_R_PORK_RE = 49.0
TAND_PORK = 0.3
EPS_R_PORK = EPS_R_PORK_RE * (1 - 1j * TAND_PORK)
MU_R_PORK = 1.0 + 0.0j
EPS_R_AIR = 1.0 + 0.0j
MU_R_AIR = 1.0 + 0.0j

# Target absorbed power for scaling (W)
TARGET_POWER_W = 700.0

# Default reference XDMF for heat/uniformity analysis
DEFAULT_REF_FILE = "cal_by_gmsh3/microwave_geometry_and_stackdumpling_0.xdmf"
_ref_file = DEFAULT_REF_FILE


def get_ref_file() -> str:
    return _ref_file


def set_ref_file(path: str) -> None:
    global _ref_file
    _ref_file = path

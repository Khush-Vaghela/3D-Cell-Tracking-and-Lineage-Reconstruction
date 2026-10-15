"""
Single source of truth for pipeline-wide constants. Every other module
imports these instead of redefining them -- the duplication across
train_detector.py, train_phase1_full.py, and test_detect.py was flagged
as a real drift risk (a hardcoded scale_down slipping out of sync with
NATIVE_SCALE/DOWNSAMPLE elsewhere was exactly this failure mode).
"""

DOWNSAMPLE: tuple[int, int, int] = (1, 4, 4)  # (Z, Y, X) strided downsample -- see preprocessing.py docstring
SIGMA_UM: float = 2.0  # still a placeholder -- see targets.py module docstring, needs measuring against real data
WINDOW_SIZE: int = 2

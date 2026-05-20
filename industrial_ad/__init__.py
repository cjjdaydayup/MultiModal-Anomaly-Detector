"""Industrial anomaly detection toolkit.

This package exposes a small Python API that can be imported by other
programs while the existing Streamlit app can continue to use the older
`core` modules.
"""

from industrial_ad.config import DetectorConfig
from industrial_ad.detector import AnomalyDetector
from industrial_ad.types import DetectionRegion, DetectionResult

__all__ = [
    "AnomalyDetector",
    "DetectorConfig",
    "DetectionRegion",
    "DetectionResult",
]

__version__ = "0.2.0"

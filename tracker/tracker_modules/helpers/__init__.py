"""
Tracker helper modules for common functionality.

This package contains reusable components that can be used by any tracker
to enhance their functionality without duplicating code.
"""

from .signal_amplifier import SignalAmplifier
from .signal_amplifier_yolo import SignalAmplifierYolo

__all__ = ['SignalAmplifier', 'SignalAmplifierYolo']
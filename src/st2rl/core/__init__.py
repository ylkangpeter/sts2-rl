# -*- coding: utf-8 -*-
"""Core module for unified state representation and action spaces"""

from .state_adapter import UnifiedState, StateAdapter
from .spaces import UnifiedActionSpace, UnifiedObservationSpace

__all__ = [
    'UnifiedState',
    'StateAdapter',
    'UnifiedActionSpace',
    'UnifiedObservationSpace',
]

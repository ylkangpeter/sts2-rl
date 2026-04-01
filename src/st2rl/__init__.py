# -*- coding: utf-8 -*-
"""Slay the Spire 2 Reinforcement Learning environment

Unified architecture supporting both headless and UI modes.
"""

__version__ = "0.2.0"

# Core
from st2rl.core import UnifiedState, StateAdapter
from st2rl.core import UnifiedActionSpace, UnifiedObservationSpace

# Environments
from st2rl.environments import UnifiedSTS2Env
from st2rl.environments import HeadlessSTS2Env
from st2rl.environments import UISTS2Env
from st2rl.environments import EnvironmentFactory

# Models
from st2rl.models import UnifiedPPOModel

# Training
from st2rl.training import UnifiedTrainer

# Inference
from st2rl.inference import UnifiedRunner

__all__ = [
    # Core
    'UnifiedState',
    'StateAdapter',
    'UnifiedActionSpace',
    'UnifiedObservationSpace',
    # Environments
    'UnifiedSTS2Env',
    'HeadlessSTS2Env',
    'UISTS2Env',
    'EnvironmentFactory',
    # Models
    'UnifiedPPOModel',
    # Training
    'UnifiedTrainer',
    # Inference
    'UnifiedRunner',
]

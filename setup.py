#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup script for st2rl"""

from setuptools import setup, find_packages

setup(
    name="sts2-rl",
    version="0.1.0",
    description="Reinforcement learning toolkit for Slay the Spire 2 built on a local sts2-cli fork",
    author="",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gymnasium==0.28.1",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "flask>=3.0.0",
        "pygame>=2.5.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "rl": [
            "stable-baselines3>=2.2.0",
            "shimmy>=0.4.0",
        ],
    },

)

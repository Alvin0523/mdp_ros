"""Shared fixtures for the mdp_bringup test suite.

Puts `mdp_bringup/scripts/` on `sys.path` so tests can import the task runners'
sibling modules (notably the pure `pose_transform` helper the frame-bug property
tests target - see README.md) the same way they resolve at runtime out of
`lib/mdp_bringup`.
"""

import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PACKAGE_ROOT / 'scripts'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(scope='session')
def package_root() -> Path:
    """Path to the mdp_bringup source package (not the install copy)."""
    return PACKAGE_ROOT


@pytest.fixture(scope='session')
def scripts_dir() -> Path:
    """Path to mdp_bringup/scripts/, home of the task runners and helpers."""
    return SCRIPTS_DIR

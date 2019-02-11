import sys
from pathlib import Path

DAVIS16_DIR = Path('/home/achald/research/misc/datasets/davis/davis-2016/')
DAVIS17_DIR = Path('/home/achald/research/misc/datasets/davis/davis-2017/')

if not DAVIS16_DIR.exists():
    raise ValueError(
        "Could not find DAVIS 2016 repo at %s. Please edit the path in %s.",
        DAVIS16_DIR, __file__)

if not DAVIS17_DIR.exists():
    raise ValueError(
        "Could not find DAVIS 2017 repo at %s. Please edit the path in %s.",
        DAVIS17_DIR, __file__)


def add_davis16_to_sys_path():
    sys.path.insert(0, str(DAVIS16_DIR / 'python' / 'lib'))


def add_davis17_to_sys_path():
    sys.path.insert(0, str(DAVIS17_DIR / 'python' / 'lib'))

import logging
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def resolve_path(x):
    if x.is_absolute():
        x = ROOT / x
    return x.resolve()


def subprocess_call(cmd, log=True, **kwargs):
    cmd = [
        str(x) if not isinstance(x, Path) else str(resolve_path(x))
        for x in cmd
    ]
    if log:
        logging.info('Command:\n%s', ' '.join(cmd).replace("--", "\\\n--"))
        if kwargs:
            logging.info('subprocess.check_call kwargs:\n%s', kwargs)
    subprocess.check_call(cmd, **kwargs)


def msg(message):
    logging.info(f'\n\n###\n{message}\n###\n')

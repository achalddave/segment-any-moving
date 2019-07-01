import logging
import subprocess


def subprocess_call(cmd, log=True):
    cmd = [str(x) for x in cmd]
    if log:
        logging.info('Command:\n%s', ' '.join(cmd).replace("--", "\\\n--"))
    subprocess.check_call(cmd)

import logging
import subprocess
from pathlib import Path


def compute_flow_helper(config, input_dir, output_dir):
    flownet2_dir = Path(config['flow']['flownet2_dir'])
    args = [
        '--input-dir', input_dir,
        '--recursive',
        '--convert-to-angle-magnitude-png', 'on',
        '--extension', '.png',
        '--gpus'] + config['general']['gpus'] + [
        '--num-workers', config['general']['num_workers'],
        '--output-dir', output_dir,
        '--flow-type', 'flownet2',
        '--flownet2-dir', flownet2_dir,
        '--flownet2-model', 'kitti'
    ]
    args = [str(x) for x in args]
    cmd = ['python', 'flow/compute_flow_sequences.py'] + args
    logging.info('Command:\n%s', ' '.join(cmd).replace("--", "\\\n--"))
    subprocess.check_call(cmd)

from pathlib import Path

from release.helpers.misc import subprocess_call


def compute_flow_helper(config, input_dir, output_dir, extension):
    flownet2_dir = Path(config['flow']['flownet2_dir'])
    args = [
        '--input-dir', input_dir,
        '--recursive',
        '--convert-to-angle-magnitude-png', 'on',
        '--extension', extension,
        '--gpus'] + config['general']['gpus'] + [
        '--num-workers', config['general']['num_workers'],
        '--output-dir', output_dir,
        '--flow-type', 'flownet2',
        '--flownet2-dir', flownet2_dir,
        '--flownet2-model', 'kitti'
    ]
    cmd = ['python', 'flow/compute_flow_sequences.py'] + args
    subprocess_call(cmd)

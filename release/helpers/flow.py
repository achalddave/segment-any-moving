from pathlib import Path

from release.helpers.misc import subprocess_call


def compute_flow_helper(config,
                        input_dir,
                        output_dir,
                        extensions=None,
                        recursive=True):
    flownet2_dir = Path(config['flow']['flownet2_dir'])

    if isinstance(extensions, str):
        extensions = [extensions]

    maybe_recursive = ['--recursive'] if recursive else []
    maybe_extensions = (['--extensions'] + extensions) if extensions else []
    args = [
        '--input-dir', input_dir
        ] + maybe_recursive + [
        '--convert-to-angle-magnitude-png', 'on'
        ] + maybe_extensions + [
        '--gpus'] + config['general']['gpus'] + [
        '--num-workers', config['general']['num_workers'],
        '--output-dir', output_dir,
        '--flow-type', 'flownet2',
        '--flownet2-dir', flownet2_dir,
        '--flownet2-model', 'kitti',
        '--quiet'
    ]
    cmd = ['python', 'flow/compute_flow_sequences.py'] + args
    subprocess_call(cmd)

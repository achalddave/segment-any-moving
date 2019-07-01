import argparse
import logging
import pprint
import subprocess
import yaml
from pathlib import Path

from script_utils.common import common_setup


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=Path('./release/config.yaml'))

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    print(config)
    fbms_config = config['fbms']
    output_dir = Path(fbms_config['flow_dir'])
    flownet2_dir = Path(config['flow']['flownet2_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    common_setup('compute_fbms_flow', output_dir, args)
    logging.debug('Config:\n%s', pprint.pformat(config))

    for split in ['test', 'train']:
        split_full = 'TrainingSet' if split == 'train' else 'TestSet'
        output_split = output_dir / split_full
        args = [
            '--input-dir', fbms_config['images_highres'][split],
            '--recursive',
            '--convert-to-angle-magnitude-png', 'on',
            '--extension', '.png',
            '--gpus'] + config['general']['gpus'] + [
            '--num-workers', config['general']['num_workers'],
            '--output-dir', output_split,
            '--flow-type', 'flownet2',
            '--flownet2-dir', flownet2_dir,
            '--flownet2-model', 'kitti'
        ]
        args = [str(x) for x in args]
        cmd = ['python', 'flow/compute_flow_sequences.py'] + args
        logging.info("\n\n"
                     "###\n"
                     "Computing flow on FBMS %s set.\n"
                     "###\n\n", split)
        logging.info('Command:\n%s', ' '.join(cmd).replace("--", "\\\n--"))
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
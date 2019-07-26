import argparse
import logging
import pprint
import subprocess
import yaml
from pathlib import Path

from script_utils.common import common_setup

from release.helpers.flow import compute_flow_helper
from release.helpers.misc import msg, subprocess_call
from utils.misc import IMG_EXTENSIONS


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default=Path('./release/config.yaml'))
    parser.add_argument('--frames-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--extensions', nargs='*', default=IMG_EXTENSIONS)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)
    logging.debug('Config:\n%s', pprint.pformat(config))

    msg(f"Computing flow on {args.frames_dir}.")
    compute_flow_helper(config,
                        args.frames_dir,
                        args.output_dir,
                        extensions=args.extensions,
                        recursive=True)


if __name__ == "__main__":
    main()
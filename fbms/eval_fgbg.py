import argparse
import logging
import subprocess
from pathlib import Path

import utils.log as log_utils

if __name__ == "__main__":
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--masks-dir', type=Path, required=True)
    parser.add_argument(
        '--fbms-dir',
        required=True,
        type=Path)
    parser.add_argument(
        '--eval-code-dir',
        type=Path,
        default='/home/achald/research/misc/fbms-fgbg-eval-pavel')
    parser.add_argument('--matlab-binary', type=Path, default='matlab')

    args = parser.parse_args()

    output_log_file = log_utils.add_time_to_path(
        args.masks_dir / (Path(__file__).name + '.log'))
    log_utils.setup_logging(output_log_file)

    for split in ['TrainingSet', 'TestSet']:
        try:
            command = [
                'matlab', '-nodesktop', '-nosplash', '-r',
                (f"evaluateAllSeqs('{args.fbms_dir}', '{args.masks_dir}', "
                 f"{{'{split}'}}); quit")
            ]
            logging.info(f'Command:\n{" ".join(command)}')
            output = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=args.eval_code_dir)
        except subprocess.CalledProcessError as e:
            logging.fatal('Failed command.\nException: %s\nOutput:\n %s',
                          e.returncode, e.output.decode('utf-8'))
            raise
        logging.info(f'{split} accuracy:')
        logging.info(output.stdout.decode('utf-8'))

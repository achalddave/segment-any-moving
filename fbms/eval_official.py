"""Evaluate predictions using FBMS evaluation code.

This is a small utility wrapper for the FBMS evaluation code."""

import argparse
import logging
import subprocess
from pathlib import Path

import utils.log as log_utils


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--eval-binary',
        help='Path to MoSegEvalAllPr binary from FBMS eval code.',
        required=True)
    parser.add_argument(
        '--predictions-dir',
        help=('Predictions directory. Should contain "all_tracks.txt" and '
              '"all_shots.txt".'))
    parser.add_argument(
        '--split', default='all', choices=['10', '50', '200', 'all'])
    parser.add_argument(
        '--object-threshold',
        default=0.75,
        help=('F-measure threshold for whether an object has been extracted. '
              'Default is 0.75 following {Ochs, Peter, Jitendra Malik, and '
              'Thomas Brox. "Segmentation of moving objects by long term '
              'video analysis." IEEE transactions on pattern analysis and '
              'machine intelligence 36.6 (2014): 1187-1200.}'))

    args = parser.parse_args()

    binary = args.eval_binary
    predictions_dir = Path(args.predictions_dir)

    logging_path = log_utils.add_time_to_path(
        predictions_dir / (Path(__file__).name + '.log'))
    log_utils.setup_logging(logging_path)
    logging.info('Args:\n%s', vars(args))

    file_logger = logging.getLogger(str(logging_path))

    shots_file = predictions_dir / 'all_shots.txt'
    tracks_file = predictions_dir / 'all_tracks.txt'
    command = [
        str(binary),
        str(shots_file), args.split,
        str(tracks_file),
        str(args.object_threshold)
    ]
    logging.info('Running command:\n%s', (' '.join(command)))

    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        file_logger.info('Output from FBMS evaluation:\n%s',
                         output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        logging.error('Error found when evaluating:')
        output = e.output.decode('utf-8')
        logging.exception(e.output.decode('utf-8'))
        if ('Could not find "Average region density" in the file!' in output
                and len(args.predictions_dir) > 250):
            logging.info(
                "\n"
                "############\n"
                "### NOTE ###\n"
                "############\n"
                "This may be due to the very long path to --predictions-dir. "
                "The FBMS evaluation code only reads up to 300 characters per "
                "line from all_tracks.txt. Either move your results to have "
                "fewer characters in the path, or edit the "
                "MoSegEvalAll_PR.cpp, run the following sed commands:\n"
                "   sed -i'' -e 's/char dummy\[300\]/char dummy\[99999\]/' MoSegEvalAll_PR.cpp\n"
                "   sed -i'' -e 's/aResult.getline(dummy,300);/aResult.getline(dummy,99999);/' MoSegEvalAll_PR.cpp"
            )
        import sys
        sys.exit(e)

    # Format of the output file from FBMS evaluation code.
    output_file = tracks_file.with_name(
        tracks_file.stem + '_Fgeq{:4.2f}'.format(100 * args.object_threshold) +
        'Numbers.txt')
    if output_file.exists():
        logging.info('Final results:\n')
        with open(output_file, 'r') as f:
            logging.info(f.read())
    else:
        logging.error(
            "Couldn't find FBMS evaluation results at path: %s" % output_file)


if __name__ == "__main__":
    main()

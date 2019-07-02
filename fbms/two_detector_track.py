"""Use two detectors: one to init and continue tracks, one only to continue.

The primary usage is to use a moving object detector (i.e. one that ignores
static objects) to initialize tracks, and a static object detector (one that
detects static and moving objects) for continuing tracks."""

import argparse
import logging
import pickle
import pprint
import subprocess
from pathlib import Path

import utils.log as log_utils
from fbms.track_all import track_fbms
from tracker import track as tracker
from tracker.two_detector_track import merge_detections
from utils.fbms import utils as fbms_utils


def main():
    tracking_parser = tracker.create_tracking_parser(suppress_args=[
        '--score-init-min', '--score-continue-min'])

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        parents=[tracking_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--init-detections-dir',
        type=Path,
        help=('Contains subdirectories for each FBMS sequence, each of which '
              'contain pickle files of detections for each frame. These '
              'detections are used to initialize and continue tracks.'),
        required=True)
    parser.add_argument(
        '--continue-detections-dir',
        type=Path,
        help=('Contains subdirectories for each FBMS sequence, each of which '
              'contain pickle files of detections for each frame. These '
              'detections are used only to continue tracks.'),
        required=True)
    parser.add_argument(
        '--remove-continue-overlap',
        type=float,
        default=0.1,
        help=('Remove detections from --continue-detections-dir if they '
              'overlap more than this threshold with a detection from '
              '--init-detections-dir.'))
    parser.add_argument(
        '--fbms-split-root',
        type=Path,
        required=True,
        help=('Directory containing subdirectories for each sequence, each of '
              'which contains a ".dat" file of groundtruth. E.g. '
              '<FBMS_ROOT>/TestSet'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--score-init-min',
        type=float,
        default=0.9,
        help=('Detection confidence threshold for starting a new track from '
              '--init-detections-dir detections.'))
    parser.add_argument(
        '--score-continue-min',
        type=float,
        default=0.7,
        help=('Detection confidence threshold for continuing a new track from '
              '--init-detections-dir or --continue-detections-dir '
              'detections.'))
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--extension', default='.jpg')
    parser.add_argument(
        '--vis-dataset',
        default='objectness',
        choices=['coco', 'objectness'],
        help='Dataset to use for mapping label indices to names.')
    parser.add_argument('--save-images', action='store_true')
    parser.add_argument('--filter-sequences', default=[], nargs='*', type=str)
    parser.add_argument('--save-merged-detections', action='store_true')

    tracking_params, remaining_argv = tracking_parser.parse_known_args()
    args = parser.parse_args(remaining_argv)

    tracking_params = vars(tracking_params)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_log_file = log_utils.add_time_to_path(
        args.output_dir / 'tracker.log')
    log_utils.setup_logging(output_log_file)
    logging.info('Args: %s', pprint.pformat(vars(args)))
    logging.info('Tracking params: %s', pprint.pformat(tracking_params))

    subprocess.call([
        './git-state/save_git_state.sh',
        output_log_file.with_suffix('.git-state')
    ])
    if args.save_merged_detections:
        output_merged = args.output_dir / 'merged'
        assert not output_merged.exists()
        output_merged.mkdir()

    # We want to use init_detections with score > s_i to init tracks, and
    # (init_detections or continue_detections with score > s_c) to continue
    # tracks. However, tracker.track only wants one set of detections, so we
    # do some score rescaling and then merge the detections.
    #
    # Let s_i be --score-init-min and s_c be --score-continue-min.
    #
    # Detections that can init tracks:
    #   I1: init_detections     with score > s_i
    #
    # Detections that can continue tracks:
    #   C1: init_detections     with score > s_c
    #   C2: continue_detections with score > s_c
    #
    # Set the score_init_min passed to the tracker to 1. Then, we can increase
    # the score of all detections in I1 to be above 1 (by adding 1.001 to the
    # score), and leave all other detections' scores as they are. 1.001 is
    # arbitrary; we just need it to be higher than any regular scoring
    # detection.
    tracking_params['score_init_min'] = 1.001
    tracking_params['score_continue_min'] = args.score_continue_min

    def detections_loader(sequence):
        init_detections = tracker.load_detectron_pickles(
            args.init_detections_dir / sequence, fbms_utils.get_framenumber)
        continue_detections = tracker.load_detectron_pickles(
            args.continue_detections_dir / sequence,
            fbms_utils.get_framenumber)
        merged_detections = merge_detections(
            init_detections,
            continue_detections,
            score_init_min=args.score_init_min,
            score_continue_min=args.score_continue_min,
            remove_continue_overlap=args.remove_continue_overlap,
            new_score_init_min=tracking_params['score_init_min'])
        if args.save_merged_detections:
            output_merged_sequence = output_merged / sequence
            output_merged_sequence.mkdir()
            for file in merged_detections:
                with open(output_merged_sequence / (file + '.pickle'),
                          'wb') as f:
                    pickle.dump(merged_detections[file], f)
        return merged_detections

    if not args.filter_sequences:
        args.filter_sequences = None

    track_fbms(
        fbms_split_root=args.fbms_split_root,
        detections_loader=detections_loader,
        output_dir=args.output_dir,
        tracking_params=tracking_params,
        frame_extension=args.extension,
        save_video=args.save_video,
        vis_dataset=args.vis_dataset,
        fps=args.fps,
        save_images=args.save_images,
        filter_sequences=args.filter_sequences,
        duplicate_last_frame=False)


if __name__ == "__main__":
    main()

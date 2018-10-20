"""Use two detectors: one to init and continue tracks, one only to continue.

The primary usage is to use a moving object detector (i.e. one that ignores
static objects) to initialize tracks, and a static object detector (one that
detects static and moving objects) for continuing tracks."""

import argparse
import logging
import pprint
import subprocess
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

import utils.log as log_utils
from fbms.track_all import track_fbms
from tracker import track as tracker
from utils.fbms import utils as fbms_utils


def filter_scores(detection, threshold):
    output = {'boxes': [], 'segmentations': [], 'keypoints': []}
    for c, boxes in enumerate(detection['boxes']):
        selected = np.where(boxes[:, 4] > threshold)[0]
        output['boxes'].append(boxes[selected])
        output['segmentations'].append(
            [detection['segmentations'][c][i] for i in selected])
        if detection['keypoints'][c]:
            output['keypoints'].append(
                [detection['keypoints'][c][i] for i in selected])
        else:
            output['keypoints'].append([])
    return output


def filter_overlapping(filter_detections, detections):
    """Remove detections in `detections` overlapping with filter_detections.

    The syntax matches that of python's `filter`, so that the second argument
    is the container to filter from.

    Args:
        from_detections (dict): Detections to filter from.
        filter_detections (dict): Detections to search for overlap with.

    Returns:
        merged (Result)
    """
    output = {'boxes': [], 'segmentations': [], 'keypoints': []}

    for c, boxes in enumerate(detections['boxes']):
        masks = detections['segmentations'][c]
        keypoints = detections['keypoints'][c]

        filter_masks = filter_detections['segmentations'][c]

        # Shape (len(masks), len(filter_masks))
        if filter_masks:
            ious = mask_util.iou(
                masks, filter_masks, pyiscrowd=np.zeros(len(filter_masks)))
            valid_indices = [
                m for m in range(len(masks)) if np.all(ious[m, :] < 0.8)
            ]
        else:
            valid_indices = list(range(len(masks)))

        valid_boxes = boxes[valid_indices]
        valid_masks = [masks[m] for m in valid_indices]
        if keypoints:
            valid_keypoints = [keypoints[m] for m in valid_indices]
        else:
            valid_keypoints = []
        output['boxes'].append(valid_boxes)
        output['segmentations'].append(valid_masks)
        output['keypoints'].append(valid_keypoints)

    return output


def standardized_detections(detections):
    """Handle edge cases in detections structure.

    - Map 'None' keypoints to an empty list.
    - Map 'None' segmentations to an empty list.
    - Map empty "boxes" list to an empty numpy array with 5 columns and 0 rows.
    """
    new_detections = detections.copy()
    if new_detections['keypoints'] is None:
        new_detections['keypoints'] = [
            [] for _ in range(len(new_detections['boxes']))
        ]

    if new_detections['segmentations'] is None:
        new_detections['segmentations'] = [
            [] for _ in range(len(new_detections['boxes']))
        ]

    for c, boxes in enumerate(new_detections['boxes']):
        if len(boxes) == 0:
            new_detections['boxes'][c] = np.zeros((0, 5))
    return new_detections


def merge_detections(detections1, detections2):
    output = {'boxes': [], 'segmentations': [], 'keypoints': []}

    detections1 = standardized_detections(detections1)
    detections2 = standardized_detections(detections2)

    for c, boxes1 in enumerate(detections1['boxes']):
        masks1 = detections1['segmentations'][c]
        keypoints1 = detections1['keypoints'][c]

        boxes2 = detections2['boxes'][c]
        masks2 = detections2['segmentations'][c]
        keypoints2 = detections2['keypoints'][c]

        output['boxes'].append(np.vstack((boxes1, boxes2)))
        output['segmentations'].append(masks1 + masks2)
        output['keypoints'].append(keypoints1 + keypoints2)
    return output


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
        default=0.7,
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
        log_utils.add_time_to_path(args.output_dir / 'git-state')
    ])

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

        merged_detections = {}
        for file, detection in init_detections.items():
            boxes = detection['boxes']
            for category_boxes in boxes[1:]:
                scores = category_boxes[:, 4]
                scores[scores > args.score_init_min] += (
                    tracking_params['score_init_min'])
                category_boxes[:, 4] = scores

            detection = standardized_detections(detection)
            continue_file_detection = standardized_detections(
                continue_detections[file])

            continue_nonoverlapping = filter_overlapping(
                filter_scores(detection,
                              tracking_params['score_continue_min']),
                continue_file_detection)
            merged_detections[file] = merge_detections(
                detection, continue_nonoverlapping)
        return merged_detections

    if not args.filter_sequences:
        args.filter_sequences = None

    track_fbms(
        fbms_split_root=args.fbms_split_root,
        detections_loader=detections_loader,
        output_dir=args.output_dir,
        tracking_params=tracking_params,
        frame_extension=args.extension,
        save_video=True,
        vis_dataset=args.vis_dataset,
        fps=args.fps,
        save_images=args.save_images,
        filter_sequences=args.filter_sequences)


if __name__ == "__main__":
    main()

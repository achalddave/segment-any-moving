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

import numpy as np
import pycocotools.mask as mask_util
from tqdm import tqdm

import utils.log as log_utils
from fbms.track_all import track_fbms
from tracker import track as tracker
from utils.fbms import utils as fbms_utils
from utils.detectron_outputs import standardized_detections


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


def filter_overlapping(filter_detections, detections, overlap_threshold):
    """Remove detections in `detections` overlapping with filter_detections.

    The syntax matches that of python's `filter`, so that the second argument
    is the container to filter from.

    Args:
        from_detections (dict): Detections to filter from.
        filter_detections (dict): Detections to search for overlap with.
        overlap_threshold (float)

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
                m for m in range(len(masks))
                if np.all(ious[m, :] < overlap_threshold)
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


def merge_detections_simple(detections1, detections2):
    output = {'boxes': [], 'segmentations': [], 'keypoints': []}

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


def merge_detections(init_detections,
                     continue_detections,
                     score_init_min,
                     score_continue_min,
                     remove_continue_overlap,
                     new_score_init_min=1.001):
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
    merged_detections = {}
    files = set(init_detections.keys()) | set(continue_detections.keys())
    for file in tqdm(files, desc='Merging'):
        if file in init_detections:
            init_detection = init_detections[file]
            boxes = init_detection['boxes']
            for category_boxes in boxes[1:]:
                scores = category_boxes[:, 4]
                scores[scores > score_init_min] += new_score_init_min
                category_boxes[:, 4] = scores

            init_detection = standardized_detections(init_detection)
            continue_file_detection = standardized_detections(
                continue_detections[file])

            continue_nonoverlapping = filter_overlapping(
                filter_scores(init_detection, score_continue_min),
                continue_file_detection,
                overlap_threshold=remove_continue_overlap)
            merged_detections[file] = merge_detections_simple(
                init_detection, continue_nonoverlapping)
        else:
            merged_detections[file] = standardized_detections(
                continue_detections[file])
    return merged_detections


def main():
    tracking_parser = tracker.create_tracking_parser(suppress_args=[
        '--score-init-min', '--score-continue-min'])

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        parents=[tracking_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument(
        '--init-detections-dir',
        type=Path,
        help=('Contains pickle files of detections for each frame.These '
              'detections are used to initialize and continue tracks.'),
        required=True)
    parser.add_argument(
        '--continue-detections-dir',
        type=Path,
        help=('Contains pickle files of detections for each frame.These '
              'detections are used only to continue tracks.'),
        required=True)
    parser.add_argument(
        '--remove-continue-overlap',
        type=float,
        default=0.1,
        help=('Remove detections from --continue-detections-dir if they '
              'overlap more than this threshold with a detection from '
              '--init-detections-dir.'))
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
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--extension', default='.jpg')
    parser.add_argument(
        '--vis-dataset',
        default='objectness',
        choices=['coco', 'objectness'],
        help='Dataset to use for mapping label indices to names.')
    parser.add_argument('--save-images', action='store_true')
    parser.add_argument('--save-merged-detections', action='store_true')
    parser.add_argument(
        '--filename-format',
        choices=[
            'frame', 'frameN', 'sequence_frame', 'sequence-frame', 'fbms'
        ],
        default='frame',
        help=('Specifies how to get frame number from the filename. '
              '"frame": the filename is the frame number, '
              '"frameN": format <frame><number>, '
              '"sequence_frame": frame number is separated by an underscore, '
              '"sequence-frame": frame number is separated by a dash, '
              '"fbms": assume fbms style frame numbers'))

    tracking_params, remaining_argv = tracking_parser.parse_known_args()
    args = parser.parse_args(remaining_argv)

    tracking_params = vars(tracking_params)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    output_log_file = log_utils.add_time_to_path(
        args.output_dir / 'tracker.log')
    log_utils.setup_logging(output_log_file)
    logging.info('Args:\n%s', pprint.pformat(vars(args)))
    logging.info('Tracking params:\n%s', pprint.pformat(tracking_params))

    subprocess.call([
        './git-state/save_git_state.sh',
        output_log_file.with_suffix('.git-state')
    ])
    if args.save_merged_detections:
        output_merged = args.output_dir / 'merged'
        assert not output_merged.exists()
        output_merged.mkdir()

    if args.filename_format == 'fbms':
        get_framenumber = fbms_utils.get_framenumber
    elif args.filename_format == 'frameN':
        def get_framenumber(x):
            return int(x.split('frame')[1])
    elif args.filename_format == 'sequence-frame':
        def get_framenumber(x):
            return int(x.split('-')[-1])
    elif args.filename_format == 'sequence_frame':
        def get_framenumber(x):
            return int(x.split('_')[-1])
    elif args.filename_format == 'frame':
        get_framenumber = int
    else:
        raise ValueError(
            'Unknown --filename-format: %s' % args.filename_format)

    tracking_params['score_init_min'] = 1.001
    tracking_params['score_continue_min'] = args.score_continue_min

    init_detections = tracker.load_detectron_pickles(
        args.init_detections_dir, get_framenumber)
    continue_detections = tracker.load_detectron_pickles(
        args.continue_detections_dir,
        get_framenumber)

    merged_detections = merge_detections(
        init_detections,
        continue_detections,
        args.score_init_min,
        args.score_continue_min,
        args.remove_continue_overlap,
        new_score_init_min=tracking_params['score_init_min'])

    if args.save_merged_detections:
        for file in merged_detections:
            with open(output_merged / (file + '.pickle'), 'wb') as f:
                pickle.dump(merged_detections[file], f)

    output_video = args.output_dir / 'video.mp4'
    output_images_dir = None
    if args.save_images:
        output_images_dir = args.output_dir / 'images'
    tracker.track_and_visualize(
        merged_detections,
        args.images_dir,
        tracking_params,
        get_framenumber,
        args.extension,
        vis_dataset='objectness',
        output_images_dir=output_images_dir,
        output_video=output_video,
        output_video_fps=args.fps,
        output_track_file=None)


if __name__ == "__main__":
    main()

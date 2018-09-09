"""Playground for tracking objects."""

import argparse
import collections
import logging
import os
import pickle
import pprint
from pathlib import Path

import cv2
import numpy as np
import PIL
import pycocotools.mask as mask_util
import skimage.color
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
from scipy.spatial.distance import cosine

import utils.vis as vis
from utils.colors import colormap
from utils.datasets import get_classes
from utils.distance import chi_square_distance, intersection_distance
from utils.log import setup_logging

# Detection confidence threshold for starting a new track
NEW_TRACK_THRESHOLD = 0.9
# Detection confidence threshold for adding a detection to existing track.
CONTINUE_TRACK_THRESHOLD = 0.5

# How many frames a track is allowed to miss detections in.
MAX_SKIP = 30

SPATIAL_THRESHOLD = 0.00005
AREA_RATIO_THRESHOLD = 0.5
IOU_GAP = 0.3
MIN_IOU = 0
APPEARANCE_FEATURE = 'histogram'  # one of 'mask' or 'histogram'
assert APPEARANCE_FEATURE in ('mask', 'histogram')
if APPEARANCE_FEATURE == 'mask':
    APPEARANCE_GAP = 0.5
else:
    APPEARANCE_GAP = 0.0


def decay_weighted_mean(values, sigma=5):
    """Weighted mean that focuses on most recent values.

    values[-i] is weighted by np.exp(i / sigma). Higher sigma weights older
    values more heavily.

    values (array-like): Values arranged from oldest to newest. values[-1] is
        weighted most heavily, values[0] is weighted least.
    """
    weights = np.exp([-i / sigma for i in range(len(values))])[::-1]
    return np.average(values, weights=weights)


class Detection():
    def __init__(self,
                 box,
                 score,
                 label,
                 timestamp,
                 image=None,
                 mask=None,
                 mask_feature=None):
        self.box = box  # (x1, y1, x2, y2)
        self.score = score
        self.label = label
        self.timestamp = timestamp
        self.image = image
        self.mask = mask
        self.mask_feature = mask_feature
        self.track = None
        self._cached_values = {
            'contour_moments': None,
            'center_box': None,
            'center_mask': None,
            'decoded_mask': None,
        }

    def contour_moments(self):
        if self._cached_values['contour_moments'] is None:
            self._cached_values['contour_moments'] = cv2.moments(
                self.decoded_mask())
        return self._cached_values['contour_moments']

    def compute_center(self):
        if self._cached_values['center_mask'] is None:
            moments = self.contour_moments()
            # See
            # <https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html>
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            self._cached_values['center_mask'] = (cx, cy)
        return self._cached_values['center_mask']

    def compute_center_box(self):
        if self._cached_values['center_box'] is None:
            x0, y0, x1, y1 = self.box
            self._cached_values['center_box'] = ((x0 + x1) / 2,
                                                   (y0 + y1) / 2)
        return self._cached_values['center_box']

    def compute_area(self):
        return self.contour_moments()['m00']

    def compute_area_bbox(self):
        x0, y0, x1, y1 = self.box
        return (x1 - x0) * (y1 - y0)

    def decoded_mask(self):
        if self._cached_values['decoded_mask'] is None:
            self._cached_values['decoded_mask'] = mask_util.decode(self.mask)
        return self._cached_values['decoded_mask']

    def bbox_centered_mask(self):
        """Return mask with respect to bounding box coordinates."""
        x1, y1, x2, y2 = self.box
        x1, y1, x2, y2 = int(x1-0.5), int(y1-0.5), int(x2+0.5), int(y2+0.5)
        return self.decoded_mask()[x1:x2+1, y1:y2+1]

    def centered_mask_iou(self, detection):
        mask = self.bbox_centered_mask()
        other_mask = detection.bbox_centered_mask()
        if mask.size == 0 or other_mask.size == 0:
            return 0
        if mask.shape != other_mask.shape:
            mask_image = PIL.Image.fromarray(mask)
            other_mask_image = PIL.Image.fromarray(other_mask)
            if mask.size > other_mask.size:
                other_mask_image = other_mask_image.resize(
                    mask_image.size, resample=PIL.Image.NEAREST)
            else:
                mask_image = mask_image.resize(
                    other_mask_image.size, resample=PIL.Image.NEAREST)
            mask = np.array(mask_image)
            other_mask = np.array(other_mask_image)
        intersection = (mask & other_mask).sum()
        union = (mask | other_mask).sum()
        if union == 0:
            return 0
        else:
            return intersection / union

    def mask_iou(self, detection):
        return mask_util.iou(
            [self.mask], [detection.mask], pyiscrowd=np.zeros(1)).item()

    def compute_histogram(self):
        # Compute histogram in LAB space, as in
        #     Xiao, Jianxiong, et al. "Sun database: Large-scale scene
        #     recognition from abbey to zoo." Computer vision and pattern
        #     recognition (CVPR), 2010 IEEE conference on. IEEE, 2010.
        # https://www.cc.gatech.edu/~hays/papers/sun.pdf
        if 'mask_histogram' in self._cached_values:
            return (self._cached_values['mask_histogram'],
                    self._cached_values['mask_histogram_edges'])
        # (num_pixels, num_channels)
        mask_pixels = self.image[np.nonzero(self.decoded_mask())]
        # rgb2lab expects a 3D tensor with colors in the last dimension, so
        # just add a fake first dimension.
        mask_pixels = mask_pixels[np.newaxis]
        mask_pixels = skimage.color.rgb2lab(mask_pixels)[0]
        # TODO(achald): Check if the range for LAB is correct.
        mask_histogram, mask_histogram_edges = np.histogramdd(
            mask_pixels,
            bins=[4, 14, 14],
            range=[[0, 100], [-127, 128], [-127, 128]])
        normalizer = mask_histogram.sum()
        if normalizer == 0:
            normalizer = 1
        mask_histogram /= normalizer
        self._cached_values['mask_histogram'] = mask_histogram
        self._cached_values['mask_histogram_edges'] = mask_histogram_edges
        return (self._cached_values['mask_histogram'],
                self._cached_values['mask_histogram_edges'])


class Track():
    def __init__(self, track_id):
        self.detections = []
        self.id = track_id
        self.velocity = None

    def add_detection(self, detection, timestamp):
        detection.track = self
        self.detections.append(detection)

    def last_timestamp(self):
        if self.detections:
            return self.detections[-1].timestamp
        else:
            return None


def track_distance(track, detection):
    # if len(track.detections) > 2:
    #     history = min(len(track.detections) - 1, 5)
    #     centers = np.asarray(
    #         [x.compute_center_box() for x in track.detections[-history:]])
    #     velocities = centers[1:] - centers[:-1]
    #     predicted_box = track.detections[-1].box
    #     if np.all(
    #             np.std(velocities, axis=0) < 0.1 *
    #             track.detections[-1].compute_area_bbox()):
    #         track.velocity = np.mean(velocities, axis=0)
    #         predicted_center = centers[-1] + track.velocity
    #     else:
    #         predicted_center = centers[-1]
    # else:
    #     predicted_box = track.detections[-1].box
    #     predicted_center = track.detections[-1].compute_center_box()
    # area = decay_weighted_mean([x.compute_area_bbox() for x in track.detections])
    # target_area = detection.compute_area_bbox()
    # return (max([abs(p1 - p0)
    #              for p0, p1 in zip(predicted_box, detection.box)]) / area)
    predicted_cx, predicted_cy = track.detections[-1].compute_center_box()
    detection_cx, detection_cy = detection.compute_center_box()
    diff_norm = ((predicted_cx - detection_cx)**2 +
                 (predicted_cy - detection_cy)**2)**0.5
    area = detection.image.shape[0] * detection.image.shape[1]
    return (diff_norm / area)


def match_detections(tracks, detections):
    """
    Args:
        track (list): List of Track objects, containing tracks up to this
            frame.
        detections (list): List of Detection objects for the current frame.

    Returns:
        matched_tracks (list): List of Track objects or None, of length
            len(detection), containing the Track, if any, that each Detection
            is assigned to.
    """
    matched_tracks = [None for _ in detections]
    # Tracks sorted by most recent to oldest.
    tracks = sorted(
        tracks,
        key=lambda t: (t.last_timestamp(), t.detections[-1].score),
        reverse=True)
    sorted_indices = sorted(
        range(len(detections)),
        key=lambda index: detections[index].score,
        reverse=True)

    candidates = {track.id: sorted_indices.copy() for track in tracks}

    # Stage 1: Keep candidates with similar areas only.
    for track in tracks:
        track_area = track.detections[-1].compute_area()
        if track_area == 0:
            candidates[track.id] = 0
            continue
        new_candidates = []
        for i in candidates[track.id]:
            d = detections[i]
            area = d.compute_area()
            if (area > 0 and (area / track_area) > AREA_RATIO_THRESHOLD
                    and (track_area / area) > AREA_RATIO_THRESHOLD):
                new_candidates.append(i)
        candidates[track.id] = new_candidates

    # Stage 2: Match tracks to detections with good mask IOU.
    for track in tracks:
        track_ious = {}
        track_detection = track.detections[-1]
        for i in candidates[track.id]:
            already_matched = matched_tracks[i] is not None
            different_label = track_detection.label != detections[i].label
            too_far = (track_distance(track, detections[i]) >
                       SPATIAL_THRESHOLD)
            if already_matched or different_label or too_far:
                continue
            track_ious[i] = track_detection.mask_iou(detections[i])
        if not track_ious:
            continue

        sorted_ious = sorted(
            track_ious.items(), key=lambda x: x[1], reverse=True)
        best_index, best_iou = sorted_ious[0]
        second_best_iou = sorted_ious[1][1] if len(sorted_ious) > 1 else 0
        if (best_iou - second_best_iou) > IOU_GAP:
            matched_tracks[best_index] = track
        else:
            candidates[track.id] = [
                d for d, iou in sorted_ious if iou >= MIN_IOU
            ]

    # Stage 3: Match tracks to detections with good appearance threshold.
    for track in tracks:
        candidates[track.id] = [
            i for i in candidates[track.id] if matched_tracks[i] is None
        ]
        if not candidates[track.id]:
            continue
        appearance_distances = {}
        track_detection = track.detections[-1]
        for i in candidates[track.id]:
            if APPEARANCE_FEATURE == 'mask':
                appearance_distances[i] = cosine(track_detection.mask_feature,
                                                 detections[i].mask_feature)
            elif APPEARANCE_FEATURE == 'histogram':
                appearance_distances[i] = chi_square_distance(
                        track_detection.compute_histogram()[0],
                        detections[i].compute_histogram()[0])
        sorted_distances = sorted(
            appearance_distances.items(), key=lambda x: x[1])
        best_match, best_distance = sorted_distances[0]
        second_best_distance = (sorted_distances[1][1] if
                                len(sorted_distances) > 1 else np.float('inf'))
        if (second_best_distance - best_distance) > APPEARANCE_GAP:
            matched_tracks[best_match] = track
    return matched_tracks


def visualize_detections(image,
                         detections,
                         dataset,
                         box_alpha=0.7,
                         dpi=200):
    if not detections:
        return image
    label_list = get_classes(dataset)

    # Display in largest to smallest order to reduce occlusion
    boxes = np.array([x.box for x in detections])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    colors = colormap()

    image = image.astype(dtype=np.uint8)
    for i in sorted_inds:
        detection = detections[i]
        color = [int(x) for x in colors[detection.track.id % len(colors), :3]]

        x0, y0, x1, y1 = [int(x) for x in detection.box]
        cx, cy = detection.compute_center_box()
        image = vis.vis_bbox(image, (x0, y0, x1 - x0, y1 - y0), color, thick=1)
        # image = vis.vis_bbox(image, (cx - 2, cy - 2, 2, 2), color, thick=3)

        # Draw spatial distance threshold
        area = detection.image.shape[0] * detection.image.shape[1]
        # cv2.circle(
        #     image, (int(cx), int(cy)),
        #     radius=int(area * SPATIAL_THRESHOLD),
        #     thickness=1,
        #     color=color)

        # if detection.track.velocity is not None:
        #     vx, vy = detection.track.velocity
        #     # Expand for visualization
        #     vx *= 2
        #     vy *= 2
        #     logging.info(
        #         'Drawing velocity at %s' % ((cx, cy, cx + vx, cy + vy), ))
        #     cv2.arrowedLine(
        #         image, (int(cx), int(cy)), (int(cx + vx), int(cy + vy)),
        #         color=color,
        #         thickness=3,
        #         tipLength=1.0)
        # else:
        #     cv2.circle(
        #         image, (int(cx), int(cy)), radius=3, thickness=1, color=color)
        image = vis.vis_mask(
            image,
            detection.decoded_mask(),
            color=color,
            alpha=0.1,
            border_thick=3)

        label_str = '({track}) {label}: {score}'.format(
            track=detection.track.id,
            label=label_list[detection.label],
            score='{:0.2f}'.format(detection.score).lstrip('0'))
        image = vis.vis_class(image, (x0, y0 - 2), label_str)

    return image


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detectron-dir', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--output-video')
    parser.add_argument('--output-video-fps', default=10, type=float)
    parser.add_argument('--output-track-file')
    parser.add_argument('--extension', default='.png')
    parser.add_argument('--dataset', default='coco', choices=['coco'])
    parser.add_argument(
        '--filename-format', choices=['frame', 'sequence_frame', 'fbms'],
        default='frame',
        help=('Specifies how to get frame number from the filename. '
              '"frame": the filename is the frame number, '
              '"sequence_frame": the frame number is separated by an '
                                 'underscore'
              '"fbms": assume fbms style frame numbers'))

    args = parser.parse_args()
    assert (args.output_dir is not None
            or args.output_video is not None
            or args.output_track_file is not None), (
            'One of --output-dir, --output-video, or --output-track-file must '
            'be specified.')

    if args.output_track_file is not None:
        output_log_file = (
            os.path.splitext(args.output_track_file)[0] + '-tracker.log')
    elif args.output_video is not None:
        output_log_file = (
            os.path.splitext(args.output_video)[0] + '-tracker.log')
    elif args.output_dir is not None:
        output_log_file = os.path.join(args.output_dir, 'tracker.log')
    setup_logging(output_log_file)
    logging.info('Printing source code to logging file')
    with open(__file__, 'r') as f:
        logging.debug(f.read())

    logging.info('Args: %s', pprint.pformat(args))

    detectron_input = Path(args.detectron_dir)
    if not detectron_input.is_dir():
        raise ValueError(
            '--detectron-dir %s is not a directory!' % args.detectron_dir)

    if args.filename_format == 'fbms':
        from utils.fbms.utils import get_framenumber
    elif args.filename_format == 'sequence_frame':
        def get_framenumber(x):
            return int(x.split('_')[-1])
    elif args.filename_format == 'frame':
        get_framenumber = int
    else:
        raise ValueError(
            'Unknown --filename-format: %s' % args.filename_format)

    data = {}
    for x in detectron_input.glob('*.pickle'):
        if x.stem == 'merged':
            logging.info('NOTE: Ignoring merged.pickle for backward '
                         'compatibility')
            continue

        try:
            get_framenumber(x.stem)
        except ValueError:
            logging.fatal('Expected pickle files to be named <frame_id>.pickle'
                          ', found %s.' % x)
            raise

        with open(x, 'rb') as f:
            data[x.stem] = pickle.load(f)

    frames = sorted(data.keys(), key=get_framenumber)

    should_visualize = (args.output_dir is not None
                        or args.output_video is not None)
    should_output_tracks = args.output_track_file is not None
    if should_output_tracks:
        logging.info('Outputing tracks to %s', args.output_track_file)
    all_tracks = []
    current_tracks = []
    track_id = 0

    label_list = get_classes(args.dataset)
    for timestamp, image_name in enumerate(tqdm(frames)):
        image = cv2.imread(
            os.path.join(args.images_dir, image_name + args.extension))
        image = image[:, :, ::-1]  # BGR -> RGB
        image_data = data[image_name]
        boxes, masks, _, labels = vis.convert_from_cls_format(
            image_data['boxes'], image_data['segmentations'],
            image_data['keypoints'])
        mask_features = [None for _ in masks]
        if 'features' in image_data:
            # features are of shape (num_segments, d)
            mask_features = list(image_data['features'])

        detections = [
            Detection(box[:4], box[4], label, timestamp, image, mask,
                      feature) for box, mask, label, feature in zip(
                          boxes, masks, labels, mask_features)
            if box[4] > CONTINUE_TRACK_THRESHOLD  #  and label == 1  # person
        ]
        matched_tracks = match_detections(current_tracks, detections)

        continued_tracks = []
        for detection, track in zip(detections, matched_tracks):
            if track is None:
                if detection.score > NEW_TRACK_THRESHOLD:
                    track = Track(track_id)
                    all_tracks.append(track)
                    track_id += 1
                else:
                    continue

            track.add_detection(detection, timestamp)
            continued_tracks.append(track)

        continued_track_ids = set([x.id for x in continued_tracks])
        skipped_tracks = []
        for track in current_tracks:
            if track.id not in continued_track_ids and (
                    track.last_timestamp() - timestamp) < MAX_SKIP:
                skipped_tracks.append(track)

        current_tracks = continued_tracks + skipped_tracks

    if should_output_tracks:
        logging.info('Outputting tracks')
        # Map frame number to list of Detections
        filtered_tracks = []
        for track in all_tracks:
            is_person = label_list[track.detections[-1].label] == 'person'
            is_long_enough = len(track.detections) > 4
            has_high_score = any(x.score >= 0.5 for x in track.detections)
            if is_person and is_long_enough and has_high_score:
                filtered_tracks.append(track)

        detections_by_frame = collections.defaultdict(list)
        for track in filtered_tracks:
            for detection in track.detections:
                detections_by_frame[detection.timestamp].append(detection)
        assert len(detections_by_frame) > 0

        output_str = ''
        # The last three fields are 'x', 'y', and 'z', and are only used for
        # 3D object detection.
        output_line_format = (
            '{frame},{track_id},{left},{top},{width},{height},{conf},-1,-1,-1'
            '\n')
        for timestamp, frame_detections in sorted(
                detections_by_frame.items(), key=lambda x: x[0]):
            for detection in frame_detections:
                x0, y0, x1, y1 = detection.box
                width = x1 - x0
                height = y1 - y0
                output_str += output_line_format.format(
                    frame=get_framenumber(frames[timestamp]),
                    track_id=detection.track.id,
                    left=x0,
                    top=y0,
                    width=width,
                    height=height,
                    conf=detection.score,
                    x=-1,
                    y=-1,
                    z=-1)
        with open(args.output_track_file, 'w') as f:
            f.write(output_str)
        logging.info('Output tracks to %s' % args.output_track_file)

    if should_visualize:
        logging.info('Visualizing tracks')
        # Map frame number to list of Detections
        detections_by_frame = collections.defaultdict(list)
        for track in all_tracks:
            for detection in track.detections:
                detections_by_frame[detection.timestamp].append(detection)

        if args.output_video is not None:
            images = []
        for timestamp, image_name in enumerate(tqdm(frames)):
            image = cv2.imread(
                os.path.join(args.images_dir, image_name + args.extension))
            image = image[:, :, ::-1]  # BGR -> RGB
            new_image = visualize_detections(
                image, detections_by_frame[timestamp], dataset=args.dataset)

            if args.output_video is not None:
                images.append(new_image)

            if args.output_dir is not None:
                new_image = PIL.Image.fromarray(new_image)
                new_image.save(
                    os.path.join(args.output_dir, image_name + '.png'))

        if args.output_video is not None:
            clip = ImageSequenceClip(images, fps=args.output_video_fps)
            clip.write_videofile(args.output_video)


if __name__ == "__main__":
    main()

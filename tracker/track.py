"""Playground for tracking objects."""

import argparse
import collections
import logging
import math
import pickle
import pprint
import subprocess
from pathlib import Path

import cv2
import numpy as np
import PIL
import pycocotools.mask as mask_util
import skimage.color
import scipy.optimize
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip
from scipy.spatial.distance import cosine

import utils.vis as vis
from utils.colors import colormap
from utils.datasets import get_classes
from utils.distance import chi_square_distance  # , intersection_distance
from utils.log import setup_logging


class Detection():
    __next_id = 0

    def __init__(self,
                 box,
                 score,
                 label,
                 timestamp,
                 image=None,
                 mask=None,
                 mask_feature=None):
        """
        Args:
            box (tuple): (x1, y1, x2, y2)
            score (float)
            label (int)
            timestamp (int)
            image (np.ndarray)
            mask (rles): Masks as encoded by pycocotools.
            mask_feature (np.ndarray): 1d array consisting of features for this
                detection.
        """
        self.box = box  # (x1, y1, x2, y2)
        self.score = score
        self.label = label
        self.timestamp = timestamp
        self.image = image
        self.mask = mask
        self.mask_feature = mask_feature
        self.track = None
        self.id = Detection.__next_id
        Detection.__next_id += 1

        self._cached_values = {
            'contour_moments': None,
            'center_box': None,
            'center_mask': None,
            'decoded_mask': None,
            'mask_histogram': None,
            'mask_histogram_edges': None
        }

    def tracked(self):
        return self.track is not None

    def clear_cache(self):
        for key in self._cached_values:
            self._cached_values[key] = None

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
            self._cached_values['center_box'] = ((x0 + x1) / 2, (y0 + y1) / 2)
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
        if self._cached_values['mask_histogram'] is not None:
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

    def normalized_box(self):
        x0, y0, x1, y1 = self.box
        h, w, _ = self.image.shape
        return [x0 / w, y0 / h, x1 / w, y1 / h]

    def __str__(self):
        output = (
            '{'
            f't: {self.timestamp}, box: {self.box}, score: {self.score}, '
            f'label: {self.label}, id: {self.label}')
        if self.tracked():
            output += f', track: {self.track}'
        return output + '}'


class Track():
    __next_id = 0

    def __init__(self, friendly_id=None):
        self.detections_by_time = {}
        self._cache = {
            'ordered_detections': None
        }

        self.velocity = None
        self.id = Track.__next_id
        # Id used for user-facing things, like visualization or outputting to a
        # text file. The id field can be very large if many tracks were created
        # then destroyed, whereas the friendly id can start at 0 and be
        # monotonically increasing and contiguous for only the tracks that are
        # visualized.
        self.friendly_id = friendly_id
        Track.__next_id += 1

    @property
    def detections(self):
        if self._cache['ordered_detections'] is None:
            self._cache['ordered_detections'] = [
                self.detections_by_time[x]
                for x in sorted(self.detections_by_time)
            ]
        return self._cache['ordered_detections']

    def nearest_detection(self, t, backward):
        """Return the nearest detection at time t for matching to.

        Args:
            backward (bool): If True, return the nearest detection with
                timestamp > t. Otherwise (by default), return the previous
                nearest detection.
        """
        if not backward:
            return self.previous_detection(t)
        else:
            return self.next_detection(t)

    def next_detection(self, t):
        """Return the next detection closest to time t."""
        try:
            return next(x for x in self.detections if x.timestamp >= t)
        except StopIteration:
            return None

    def previous_detection(self, t):
        """Return the previous detection closest to time t."""
        try:
            return next(
                x for x in reversed(self.detections) if x.timestamp <= t)
        except StopIteration:
            return None

    def add_detection(self, detection):
        assert detection.track is None or detection.track == self, (
            'detection already assigned to another track')
        t = detection.timestamp
        assert t not in self.detections_by_time, (
            'Track already has detection at time t=%s' % t)
        detection.track = self
        self.detections_by_time[t] = detection
        self._cache['ordered_detections'] = None

    def __str__(self):
        output = f'{{id: {self.id}'
        if self.friendly_id:
            output += f', friendly_id: {self.friendly_id}'
        if self.detections:
            output += ', time range: ({start}, {end})'.format(
                start=self.detections[0].timestamp,
                end=self.detections[-1].timestamp)
        return output + '}'


def track_distance(track, detection, backward):
    predicted_cx, predicted_cy = track.nearest_detection(
        detection.timestamp, backward).compute_center_box()
    detection_cx, detection_cy = detection.compute_center_box()
    diff_norm = ((predicted_cx - detection_cx)**2 +
                 (predicted_cy - detection_cy)**2)**0.5
    diagonal = (detection.image.shape[0]**2 + detection.image.shape[1]**2)**0.5
    return (diff_norm / diagonal)


def filter_areas(tracks,
                 candidates,
                 area_ratio,
                 backward):
    """
    Args:
        tracks (list): List of Tracks
        candidates (dict): Map track id to list of Detections.

    Returns:
        candidates (dict): Map track id to list of Detections.
    """
    new_candidates = {}
    timestamp = None
    for track in tracks:
        if not candidates[track.id]:
            continue
        if timestamp is None:
            timestamp = candidates[track.id][0]
        nearest_detection = track.nearest_detection(timestamp, backward)
        track_area = nearest_detection.compute_area()
        new_candidates[track.id] = []
        if track_area == 0:
            continue
        for d in candidates[track.id]:
            area = d.compute_area()
            if (area > 0
                    and ((1 / area_ratio) > (area / track_area) > area_ratio)):
                new_candidates[track.id].append(d)
    return new_candidates


def filter_labels(tracks, candidates):
    """Filter candidates with different labels."""
    new_candidates = {}
    for track in tracks:
        assert all(
            x.label == track.detections[0].label for x in track.detections[1:])
        track_label = track.detections[0].label
        new_candidates[track.id] = [
            d for d in candidates[track.id] if d.label == track_label
        ]
    return new_candidates


def filter_spatial_distance(tracks, candidates, max_distance, backward):
    return {
        track.id: [
            detection for detection in candidates[track.id]
            if track_distance(track, detection, backward) < max_distance
        ]
        for track in tracks
    }


def compute_ious(tracks, candidates, backward):
    """
    Returns:
        ious (dict): Map track.id to list of ious for each candidate detection
            in new_candidates.
    """
    ious = {}

    for track in tracks:
        if not candidates[track.id]:
            continue
        ious[track.id] = [
            track.nearest_detection(detection.timestamp,
                                    backward).mask_iou(detection)
            for detection in candidates[track.id]
        ]
    return ious


def filter_appearance(tracks,
                      candidates,
                      appearance_feature,
                      appearance_gap,
                      backward):
    new_candidates = {}
    for track in tracks:
        if not candidates[track.id]:
            continue
        appearance_distances = {}
        track_detection = track.nearest_detection(
            candidates[track.id][0].timestamp, backward)
        for i, detection in enumerate(candidates[track.id]):
            if appearance_feature == 'mask':
                appearance_distances[i] = cosine(track_detection.mask_feature,
                                                 detection.mask_feature)
            elif appearance_feature == 'histogram':
                appearance_distances[i] = chi_square_distance(
                    track_detection.compute_histogram()[0],
                    detection.compute_histogram()[0])

        sorted_distances = sorted(
            appearance_distances.items(), key=lambda x: x[1])
        best_match, best_distance = sorted_distances[0]
        second_best_distance = (sorted_distances[1][1] if
                                len(sorted_distances) > 1 else np.float('inf'))
        if ((second_best_distance - best_distance) > appearance_gap):
            new_candidates[track.id] = [candidates[track.id][best_match]]
        else:
            new_candidates[track.id] = candidates[track.id]
    return new_candidates


def match_detections_single_timestep(tracks,
                                     detections,
                                     tracking_params,
                                     backward):
    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    if not detections:
        return {}
    timestamp = detections[0].timestamp

    tracks = sorted(
        tracks,
        key=lambda track: track.nearest_detection(timestamp, backward).score,
        reverse=True)
    candidates = {track.id: list(detections) for track in tracks}

    # Keep candidates with similar areas only.
    if tracking_params['area_ratio'] > 0:
        candidates = filter_areas(tracks, candidates,
                                  tracking_params['area_ratio'], backward)

    if tracking_params['spatial_dist_max'] > 0:
        candidates = filter_spatial_distance(
            tracks, candidates, tracking_params['spatial_dist_max'], backward)

    # Maps track ids to iou for each track candidate
    ious = compute_ious(tracks, candidates, backward)

    # If we're not using the appearance, just use Hungarian method to assing
    # tracks and detections.
    if tracking_params['appearance_feature'] == 'none':
        ious_matrix = np.zeros((len(tracks), len(detections)))
        detection_indices = {d.id: index for index, d in enumerate(detections)}

        matched_tracks = {detection.id: None for detection in detections}
        for t, track in enumerate(tracks):
            if track.id not in ious:
                continue
            for detection, iou in zip(candidates[track.id], ious[track.id]):
                d = detection_indices[detection.id]
                ious_matrix[t, d] = iou
        # Tuple of (track indices, detection indices)
        assignments = scipy.optimize.linear_sum_assignment(-ious_matrix)
        for t, d in zip(assignments[0].tolist(), assignments[1].tolist()):
            if ious_matrix[t, d] > tracking_params['iou_min']:
                matched_tracks[detections[d].id] = tracks[t]
        return matched_tracks

    for track in tracks:
        if track.id not in candidates:
            continue
        if track.id not in ious:
            candidates[track.id] = []
            continue
        candidates[track.id] = [
            d for d, iou in zip(candidates[track.id], ious[track.id])
            if iou > tracking_params['iou_min']
        ]

    # Stage 3: Match tracks to detections with good appearance threshold.
    if tracking_params['appearance_feature'] != 'none':
        candidates = filter_appearance(tracks, candidates,
                                       tracking_params['appearance_feature'],
                                       tracking_params['appearance_gap'],
                                       backward)

        # Loop over detections in descending order of scores; if the detection
        # is the only candidate for any track, assign it to that track.
        matched_tracks = {detection.id: None for detection in detections}
        for detection in detections:
            for track in tracks:
                if track.id not in candidates:
                    continue
                track_candidates = candidates[track.id]
                if ((len(track_candidates) == 1)
                        and (track_candidates[0].id == detection.id)):
                    matched_tracks[detection.id] = track
                    del candidates[track.id]
        return matched_tracks


def match_detections(tracks, detections, tracking_params, backward):
    """
    Args:
        track (list): List of Track objects, containing tracks up to this
            frame.
        detections (list): List of Detection objects for the current frame.
        backward (bool): Whether we are tracking backwards.

    Returns:
        matched_tracks (dict): Map detection id to matching Track, or None.
    """
    t = detections[0].timestamp

    # Tracks sorted by closest to furthest in time from current time.
    tracks_by_offset = collections.defaultdict(list)
    for track in tracks:
        if backward:
            closest_offset = track.next_detection(t).timestamp - t
        else:
            closest_offset = t - track.previous_detection(t).timestamp
        tracks_by_offset[closest_offset].append(track)

    time_offsets = sorted(tracks_by_offset.keys())
    assert all(t > 0 for t in time_offsets)

    matched_tracks = {d.id: None for d in detections}

    # Match detections to the most recent tracks first.
    unmatched_detections = detections
    detections_by_id = {d.id: d for d in detections}
    for time_offset in time_offsets:
        single_timestamp_matched_tracks = match_detections_single_timestep(
            tracks_by_offset[time_offset], unmatched_detections,
            tracking_params, backward)
        new_unmatched_detections = []
        for detection_id, track in single_timestamp_matched_tracks.items():
            if track is not None:
                assert matched_tracks[detection_id] is None
                matched_tracks[detection_id] = track
            else:
                new_unmatched_detections.append(detections_by_id[detection_id])
        unmatched_detections = new_unmatched_detections

    return matched_tracks


def visualize_detections(image,
                         detections,
                         dataset,
                         tracking_params,
                         vis_bbox=False,
                         vis_label=True,
                         dpi=200):
    if not detections:
        return image
    label_list = get_classes(dataset)

    # Display in largest to smallest order to reduce occlusion
    boxes = np.array([x.box for x in detections])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    colors = colormap(rgb=True, lighten=True)

    image = image.astype(dtype=np.uint8)
    for i in sorted_inds:
        detection = detections[i]
        track_friendly_id = detection.track.friendly_id
        assert track_friendly_id is not None
        color = [int(x) for x in colors[track_friendly_id % len(colors), :3]]

        x0, y0, x1, y1 = [int(x) for x in detection.box]
        if vis_bbox:
            image = vis.vis_bbox(
                image, (x0, y0, x1 - x0, y1 - y0), color, thick=1)

        # Draw spatial distance threshold
        if tracking_params['draw_spatial_threshold']:
            cx, cy = detection.compute_center_box()
            diagonal = (
                detection.image.shape[0]**2 + detection.image.shape[1]**2)**0.5
            cv2.circle(
                image, (int(cx), int(cy)),
                radius=int(diagonal * tracking_params['spatial_dist_max']),
                thickness=1,
                color=color)

        image = vis.vis_mask(
            image,
            detection.decoded_mask(),
            color=color,
            alpha=0.5,
            border_alpha=0.5,
            border_color=[255, 255, 255],
            border_thick=3)

        if vis_label:
            label_str = '({track}) {label}: {score}'.format(
                track=track_friendly_id,
                label=label_list[detection.label],
                score='{:0.2f}'.format(detection.score).lstrip('0'))
            image = vis.vis_class(image, (x0, y0 - 2), label_str)

    return image


def track(frame_paths,
          frame_detections,
          tracking_params,
          progress=True,
          filter_label=None):
    """
    Args:
        frame_paths (list): List of paths to frames.
        frame_detections (list): List of detection results for each frame. Each
            element is a dictionary containing keys 'boxes', 'masks', and
            'keypoints'.
        tracking_params (dict): See add_tracking_arguments() for details.
        label_list (list): List of label names.
        filter_label (str):
    """
    all_tracks = []

    # detections[i] contains list of Detections for frame_paths[i]
    detections = []
    dummy_image = None
    for t, (frame_path, image_results) in enumerate(
            zip(frame_paths, frame_detections)):
        if tracking_params['appearance_feature'] == 'none':
            if dummy_image is None:
                dummy_image = np.zeros_like(cv2.imread(str(frame_path)))
            image = dummy_image
        else:
            image = cv2.imread(str(frame_path))[:, :, ::-1]  # BGR -> RGB
        boxes, masks, _, labels = vis.convert_from_cls_format(
            image_results['boxes'], image_results['segmentations'],
            image_results['keypoints'])

        if boxes is None:
            logging.info('No predictions for image %s', frame_path.name)
            boxes, masks = [], []

        if ('features' in image_results
                and tracking_params['appearance_feature'] == 'mask'):
            # features are of shape (num_segments, d)
            features = list(image_results['features'])
        else:
            features = [None for _ in masks]

        current_detections = []
        for box, mask, label, feature in zip(boxes, masks, labels, features):
            low_scoring = box[4] <= tracking_params['score_continue_min']
            label_mismatch = filter_label is not None and label != filter_label
            if low_scoring or label_mismatch:
                continue
            current_detections.append(
                Detection(box[:4], box[4], label, t, image, mask, feature))
        detections.append(current_detections)

    if tracking_params['bidirectional']:
        directions = ['forward', 'backward']
    else:
        directions = ['forward']
    for direction in directions:
        forward = direction == 'forward'
        timestamps = range(len(frame_paths))
        if not forward:
            timestamps = reversed(timestamps)

        for t in tqdm(
                timestamps,
                disable=not progress,
                total=len(frame_paths),
                desc='track ' + direction):
            frame_path = frame_paths[t]
            frame_detections = [d for d in detections[t] if not d.tracked()]
            if not frame_detections:
                continue

            for track in all_tracks:
                for detection in track.detections:
                    detection.clear_cache()

            if forward:
                active_tracks = [
                    track for track in all_tracks
                    if ((t - track.detections[-1].timestamp
                         ) <= tracking_params['frames_skip_max'])
                ]
            else:
                active_tracks = []
                for track in all_tracks:
                    # Keep tracks that
                    # (1) end at or after t,
                    # (2) start before t + frames_skip_max
                    # (3) don't have a detection at time t
                    ends_after = track.detections[-1].timestamp > t
                    starts_before_skip = (
                        track.detections[0].timestamp <
                        t + tracking_params['frames_skip_max'])
                    needs_detection = t not in track.detections_by_time
                    if ends_after and starts_before_skip and needs_detection:
                        active_tracks.append(track)
                if not active_tracks:
                    continue

            matched_tracks = match_detections(
                active_tracks,
                frame_detections,
                tracking_params,
                backward=not forward)

            # Tracks that were assigned a detection in this frame.
            for detection in frame_detections:
                track = matched_tracks[detection.id]
                if track is None:
                    if detection.score > tracking_params['score_init_min']:
                        track = Track()
                        all_tracks.append(track)
                    else:
                        continue
                track.add_detection(detection)

    for index, t in enumerate(all_tracks):
        t.friendly_id = index
    return all_tracks


def output_mot_tracks(tracks, label_list, frame_numbers, output_track_file):
    filtered_tracks = []

    for track in tracks:
        is_person = label_list[track.detections[-1].label] == 'person'
        is_long_enough = len(track.detections) > 4
        has_high_score = any(x.score >= 0.5 for x in track.detections)
        if is_person and is_long_enough and has_high_score:
            filtered_tracks.append(track)

    # Map frame number to list of Detections
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
                frame=frame_numbers[timestamp],
                track_id=detection.track.friendly_id,
                left=x0,
                top=y0,
                width=width,
                height=height,
                conf=detection.score,
                x=-1,
                y=-1,
                z=-1)
    with open(output_track_file, 'w') as f:
        f.write(output_str)


def output_numpy_tracks(tracks, output_numpy, output_numpy_every_kth,
                        num_frames, image_width, image_height):
    """
    Args:
        tracks (list of Track)
        output_numpy (Path)
        output_numpy_every_kth (int): Output only every kth frame in the numpy
            file. This is useful, for example, when a dataset has sparse
            annotations, and we only need to save the segmentations for, say,
            every 6th frame, as in YTVOS.
        num_frames (int): Number of frames in the video.
        image_width (int)
        image_height (int)
    """
    track_ids = {}  # Map track id to sequential numbers starting at 1
    # Map frame number to list of Detections
    detections_by_frame = collections.defaultdict(list)
    for track in tracks:
        for detection in track.detections:
            detections_by_frame[detection.timestamp].append(detection)
        if track.id not in track_ids:
            track_ids[track.id] = len(track_ids) + 1

    num_output_frames = math.ceil(num_frames / output_numpy_every_kth)
    full_segmentation = np.zeros((num_output_frames, image_height,
                                  image_width))

    for timestamp, frame_detections in sorted(
            detections_by_frame.items(), key=lambda x: x[0]):
        timestamp, remainder = divmod(timestamp, output_numpy_every_kth)
        if remainder != 0:
            continue
        # Paint segmentation in ascending order of confidence
        frame_detections = sorted(frame_detections, key=lambda d: d.score)
        segmentation = np.zeros((image_height, image_width))
        for detection in frame_detections:
            segmentation[detection.decoded_mask() != 0] = (
                track_ids[detection.track.id])
        full_segmentation[timestamp, :, :] = segmentation
    np.savez_compressed(output_numpy, segmentation=full_segmentation)


def visualize_tracks(tracks,
                     frame_paths,
                     dataset,
                     tracking_params,
                     output_dir=None,
                     output_video=None,
                     output_video_fps=10,
                     vis_bbox=False,
                     vis_label=False,
                     progress=False):
    """
    Args:
        tracks (list): List of Track objects
        frame_paths (list): List of frame paths.
        dataset (str): Dataset for utils/vis.py
        tracking_params (dict): Tracking parameters.
    """
    # Map frame number to list of Detections
    detections_by_frame = collections.defaultdict(list)
    for track in tracks:
        for detection in track.detections:
            detections_by_frame[detection.timestamp].append(detection)

    def visualize_image(timestamp):
        image_path = frame_paths[timestamp]
        image = cv2.imread(str(image_path))
        image = image[:, :, ::-1]  # BGR -> RGB
        new_image = visualize_detections(
            image,
            detections_by_frame[timestamp],
            dataset=dataset,
            tracking_params=tracking_params,
            vis_bbox=vis_bbox,
            vis_label=vis_label)

        return new_image

    if output_video is None:
        assert output_dir is not None
        for t, image_path in tqdm(
                enumerate(frame_paths), disable=not progress):
            image = PIL.Image.fromarray(visualize_image(t))
            image.save(output_dir / (image_path.name + '.png'))
        return

    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
    # Some videos don't play in Firefox and QuickTime if '-pix_fmt yuv420p' is
    # not specified, and '-pix_fmt yuv420p' requires that the dimensions be
    # even, so we need the '-vf scale=...' filter.
    ffmpeg_params = [
        '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2", '-pix_fmt', 'yuv420p'
    ]
    width, height = PIL.Image.open(str(frame_paths[0])).size
    with FFMPEG_VideoWriter(
            str(output_video),
            size=(width, height),
            fps=output_video_fps,
            ffmpeg_params=ffmpeg_params) as writer:
        for t, image_path in enumerate(
                tqdm(frame_paths, disable=not progress)):
            visualized = visualize_image(t)
            if output_dir is not None:
                image = PIL.Image.fromarray(visualized)
                image.save(output_dir / (image_path.name + '.png'))
            writer.write_frame(visualized)


def create_tracking_parser(suppress_args=None):
    """Create a parser with tracking arguments.

    Usage:
        tracking_parser = create_tracking_parser()

        parser = argparse.ArgumentParser(parents=[tracking_parser])

        tracking_params, remaining_argv = tracking_parser.parse_known_args()
        args = parser.parse_args(remaining_argv)

    Note that if add_tracking_params adds any required arguments, then the
    above code will cause --help to fail. Since the `tracking_parser` returned
    from this function does not implement --help, it will try to parse the
    known arguments even if --help is specified, and fail when a required
    argument is missing.
    """
    parent_parser = argparse.ArgumentParser(add_help=False)
    add_tracking_arguments(parent_parser, suppress_args)
    return parent_parser


def add_tracking_arguments(root_parser, suppress_args=None):
    """Add tracking params to an argument parser.

    Args:
        suppress_args (list): List of arguments to suppress. This can be used
            if the corresponding arguments will be populated by the calling
            script.
    """
    parser = root_parser.add_argument_group('Tracking params')
    if suppress_args is None:
        suppress_args = []

    suppressed = set()

    def add_argument(argument, parser=parser, *args, **kwargs):
        if argument in suppress_args:
            suppressed.add(argument)
            return
        parser.add_argument(argument, *args, **kwargs)

    add_argument(
        '--score-init-min',
        default=0.9,
        type=float,
        help='Detection confidence threshold for starting a new track')
    add_argument(
        '--score-continue-min',
        default=0.7,
        type=float,
        help=('Detection confidence threshold for adding a detection to '
              'existing track.'))
    add_argument(
        '--frames-skip-max',
        default=10,
        type=int,
        help='How many frames a track is allowed to miss detections in.')
    add_argument(
        '--spatial-dist-max',
        default=-1,
        type=float,
        help=('Maximum distance between matched detections, as a fraction of '
              'the image diagonal. Set to a negative value to disable '
              '(disabled by default).'))
    add_argument(
        '--area-ratio',
        default=0,
        type=float,
        help=('Specifies threshold for area ratio between matched detections. '
              'To match two detections, the ratio between their areas must '
              'be between this threshold and 1/threshold. By default this is '
              'set to 0, which disables checking area ratios.'))
    add_argument(
        '--iou-min',
        default=0.1,
        type=float,
        help='Minimum IoU between matched detections.')
    add_argument(
        '--ignore-labels',
        action='store_true',
        help=('Ignore labels when matching detections. By default, we only '
              'match detections if their labels match.'))
    parser.add_argument(
        '--bidirectional',
        action='store_true',
        help='Whether to track backwards as well as forwards in time.')

    # TODO(achald): Add help text.
    add_argument(
        '--appearance-feature',
        choices=['mask', 'histogram', 'none'],
        default='none',
        help=('Appearance feature for final comparisons between detections. '
              'This is only used if multiple detections have > --iou-min IoU '
              'with a track. By default, appearance features are not used and '
              'the Hungarian algorithm is used to make assignments based on '
              'IoU.'))
    add_argument('--appearance-gap', default=0, type=float)

    debug_parser = root_parser.add_argument_group('debug')
    add_argument(
        '--draw-spatial-threshold',
        parser=debug_parser,
        action='store_true',
        help='Draw a diagnostic showing the spatial distance threshold')

    if len(suppressed) != len(suppress_args):
        logging.warn('Unknown arguments in suppress_args: '
                     f'{suppressed-set(suppress_args)}')


def load_detectron_pickles(detectron_input, frame_parser):
    """Load detectron pickle files from a directory.

    Returns:
        dict, mapping pickle filename to data in pickle file, which should
        be a dictionary containing keys 'boxes', 'masks', and 'keypoints'.
    """
    detection_results = {}
    for x in detectron_input.glob('*.pickle'):
        if x.stem == 'merged':
            logging.warn('Ignoring merged.pickle for backward compatibility')
            continue

        try:
            frame_parser(x.stem)
        except ValueError:
            logging.fatal('Unexpected pickle file name: %s', x)
            raise

        with open(x, 'rb') as f:
            detection_results[x.stem] = pickle.load(f)
    return detection_results


def track_and_visualize(detection_results,
                        images_dir,
                        tracking_params,
                        get_framenumber,
                        frame_extension,
                        vis_dataset=None,
                        output_numpy=None,
                        output_numpy_every_kth=1,
                        output_images_dir=None,
                        output_video=None,
                        output_video_fps=None,
                        output_track_file=None,
                        progress=True):
    frames = sorted(detection_results.keys(), key=get_framenumber)

    should_visualize = (output_images_dir is not None
                        or output_video is not None)
    should_output_mot = output_track_file is not None
    should_output_numpy = output_numpy is not None
    if should_output_mot:
        logging.info('Will output MOT style tracks to %s',
                     output_track_file)
    if should_output_numpy:
        assert output_numpy.suffix == '.npz', 'output_numpy must end in .npz'

    label_list = get_classes(vis_dataset)

    frame_paths = [
        images_dir / (frame + frame_extension) for frame in frames
    ]
    all_tracks = track(frame_paths,
                       [detection_results[frame] for frame in frames],
                       tracking_params,
                       progress=progress)
    w, h = PIL.Image.open(frame_paths[0]).size

    if should_output_mot:
        logging.info('Outputting MOT style tracks')
        output_mot_tracks(all_tracks, label_list,
                          [get_framenumber(x) for x in frames],
                          output_track_file)
        logging.info('Output tracks to %s' % output_track_file)

    if should_output_numpy:
        output_numpy_tracks(all_tracks, output_numpy, output_numpy_every_kth,
                            len(frame_paths), w, h)

    if should_visualize:
        visualize_tracks(
            all_tracks,
            frame_paths,
            vis_dataset,
            tracking_params,
            output_images_dir,
            output_video,
            output_video_fps,
            progress=progress)


def main():
    tracking_parser = create_tracking_parser()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        parents=[tracking_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detectron-dir', type=Path, required=True)
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--output-images-dir', type=Path)
    parser.add_argument('--output-video', type=Path)
    parser.add_argument('--output-video-fps', default=30, type=float)
    parser.add_argument(
        '--output-track-file',
        type=Path,
        help='Optional; path to output MOT17 style tracking output.')
    parser.add_argument('--extension', default='.png')
    parser.add_argument(
        '--dataset', default='coco', choices=['coco', 'objectness'])
    parser.add_argument(
        '--filename-format',
        choices=['frame', 'sequence_frame', 'sequence-frame', 'fbms'],
        default='frame',
        help=('Specifies how to get frame number from the filename. '
              '"frame": the filename is the frame number, '
              '"sequence_frame": frame number is separated by an underscore, '
              '"sequence-frame": frame number is separated by a dash, '
              '"fbms": assume fbms style frame numbers'))

    tracking_params, remaining_argv = tracking_parser.parse_known_args()
    args = parser.parse_args(remaining_argv)

    tracking_params = vars(tracking_params)

    assert (args.output_images_dir is not None
            or args.output_video is not None
            or args.output_track_file is not None), (
            'One of --output-dir, --output-video, or --output-track-file must '
            'be specified.')

    if args.output_track_file is not None:
        output_log_file = args.output_track_file.with_name(
            args.output_track_file.stem + '-tracker.log')
    elif args.output_video is not None:
        output_log_file = args.output_video.with_name(args.output_video.stem +
                                                      '-tracker.log')
    elif args.output_images_dir is not None:
        output_log_file = args.output_images_dir / 'tracker.log'

    output_log_file.parent.mkdir(exist_ok=True, parents=True)
    setup_logging(output_log_file)
    logging.info('Printing source code to logging file')
    with open(__file__, 'r') as f:
        logging.debug(f.read())

    logging.info('Args: %s', pprint.pformat(vars(args)))
    logging.info('Tracking params: %s', pprint.pformat(tracking_params))
    subprocess.call([
        './git-state/save_git_state.sh',
        output_log_file.with_suffix('.git-state')
    ])

    detectron_input = args.detectron_dir
    if not detectron_input.is_dir():
        raise ValueError(
            '--detectron-dir %s is not a directory!' % args.detectron_dir)

    if args.filename_format == 'fbms':
        from utils.fbms.utils import get_framenumber
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

    detection_results = load_detectron_pickles(
        args.detectron_dir, frame_parser=get_framenumber)
    track_and_visualize(detection_results,
                        args.images_dir,
                        tracking_params,
                        get_framenumber,
                        args.extension,
                        vis_dataset=args.dataset,
                        output_images_dir=args.output_images_dir,
                        output_video=args.output_video,
                        output_video_fps=args.output_video_fps,
                        output_track_file=args.output_track_file)


if __name__ == "__main__":
    main()

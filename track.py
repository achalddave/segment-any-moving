"""Playground for tracking objects."""

import argparse
import os
import pickle

import cv2
import numpy as np
import PIL
import pycocotools.mask as mask_util
import skimage.color
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

import utils.vis as vis
from utils.colors import colormap
from utils.datasets import get_classes
from utils.distance import chi_square_distance, intersection_distance

# Detection confidence threshold for starting a new track
NEW_TRACK_THRESHOLD = 0.9
# Detection confidence threshold for adding a detection to existing track.
CONTINUE_TRACK_THRESHOLD = 0.5

# How many frames a track is allowed to miss detections in.
MAX_SKIP = 30

SPATIAL_DISTANCE_THRESHOLD = 0.00005
HISTOGRAM_THRESHOLD = 0.5  # Use ~1.4 for histogram intersection


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

    def contour_moments(self):
        if not hasattr(self, '_contour_moments'):
            self._contour_moments = cv2.moments(self.decoded_mask())
        return self._contour_moments

    def compute_center(self):
        moments = self.contour_moments()
        # See
        # <https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html>
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        self.center = np.asarray([cx, cy])
        return self.center

    def compute_area(self):
        return self.contour_moments()['m00']

    def compute_area_bbox(self):
        x0, y0, x1, y1 = self.box
        return (x1 - x0) * (y1 - y0)

    def decoded_mask(self):
        return mask_util.decode(self.mask)

    def compute_histogram(self):
        # Compute histogram in LAB space, as in
        #     Xiao, Jianxiong, et al. "Sun database: Large-scale scene
        #     recognition from abbey to zoo." Computer vision and pattern
        #     recognition (CVPR), 2010 IEEE conference on. IEEE, 2010.
        # https://www.cc.gatech.edu/~hays/papers/sun.pdf
        if hasattr(self, 'mask_histogram'):
            return self.mask_histogram, self.mask_histogram_edges
        # (num_pixels, num_channels)
        mask_pixels = self.image[np.nonzero(self.decoded_mask())]
        # rgb2lab expects a 3D tensor with colors in the last dimension, so
        # just add a fake dimension.
        mask_pixels = mask_pixels[np.newaxis]
        mask_pixels = skimage.color.rgb2lab(mask_pixels)[0]
        # TODO(achald): Check if the range for LAB is correct.
        self.mask_histogram, self.mask_histogram_edges = np.histogramdd(
            mask_pixels,
            bins=[4, 14, 14],
            range=[[0, 100], [-127, 128], [-127, 128]])
        normalizer = self.mask_histogram.sum()
        if normalizer == 0:
            normalizer = 1
        self.mask_histogram /= normalizer
        return self.mask_histogram, self.mask_histogram_edges


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


def compute_bbox_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def track_distance(track, detection):
    if len(track.detections) > 2:
        history = min(len(track.detections) - 1, 5)
        centers = np.asarray(
            [x.compute_center() for x in track.detections[-history:]])
        velocities = centers[1:] - centers[:-1]
        predicted_box = track.detections[-1].box
        if False and np.all(
                np.std(velocities, axis=0) < 0.1 *
                track.detections[-1].compute_area_bbox()):
            track.velocity = np.mean(velocities, axis=0)
            predicted_center = centers[-1] + track.velocity
        else:
            predicted_center = centers[-1]
    else:
        predicted_box = track.detections[-1].box
        predicted_center = track.detections[-1].compute_center()
    # area = decay_weighted_mean([x.compute_area_bbox() for x in track.detections])
    # target_area = detection.compute_area_bbox()
    # return (max([abs(p1 - p0)
    #              for p0, p1 in zip(predicted_box, detection.box)]) / area)
    return (np.linalg.norm((predicted_center - detection.compute_center()), 2)
            / detection.image.shape[0] / detection.image.shape[1])


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
    left_indices = set(range(len(detections)))
    # Tracks sorted by most recent to oldest.
    tracks = sorted(
        tracks,
        key=lambda t: (t.last_timestamp(), t.detections[-1].score),
        reverse=True)
    sorted_indices = sorted(
        range(len(detections)),
        key=lambda index: detections[index].score,
        reverse=True)
    for track in tracks:
        best_histogram_distance = np.float('inf')
        best_match = None
        track_histogram = track.detections[-1].compute_histogram()[0]
        # track_histogram = np.sum(
        #     x.compute_histogram()[0] for x in track.detections)
        # if track_histogram.sum() != 0:
        #     track_histogram = track_histogram / track_histogram.sum()
        for i in sorted_indices:
            if i not in left_indices:
                continue
            spatial_distance = track_distance(track, detections[i])
            if (spatial_distance < SPATIAL_DISTANCE_THRESHOLD):
                distance = chi_square_distance(
                    track.detections[-1].compute_histogram()[0],
                    detections[i].compute_histogram()[0])
                if distance < best_histogram_distance:
                    best_histogram_distance = distance
                    best_match = i
        if (best_match is not None
                and best_histogram_distance < HISTOGRAM_THRESHOLD):
            matched_tracks[best_match] = track
            left_indices.remove(best_match)
    return matched_tracks


def visualize_detections(image,
                         detections,
                         dataset,
                         box_alpha=0.7,
                         dpi=200):
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
        cx, cy = detection.compute_center()
        image = vis.vis_bbox(image, (x0, y0, x1 - x0, y1 - y0), color, thick=1)
        # image = vis.vis_bbox(image, (cx - 2, cy - 2, 2, 2), color, thick=3)

        # Draw spatial distance threshold
        area = detection.image.shape[0] * detection.image.shape[1]
        cv2.circle(
            image, (cx, cy),
            radius=int(area * SPATIAL_DISTANCE_THRESHOLD),
            thickness=1,
            color=color)

        if detection.track.velocity is not None:
            vx, vy = detection.track.velocity
            # Expand for visualization
            vx *= 2
            vy *= 2
            print('Drawing velocity at %s' % ((cx, cy, cx + vx, cy + vy), ))
            cv2.arrowedLine(
                image, (int(cx), int(cy)), (int(cx + vx), int(cy + vy)),
                color=color,
                thickness=3,
                tipLength=1.0)
        else:
            cv2.circle(image, (cx, cy), radius=3, thickness=1, color=color)
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
    parser.add_argument('--detectron-pickle', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--output-video')
    parser.add_argument('--output-video-fps', default=3, type=float)
    parser.add_argument('--extension', default='.png')
    parser.add_argument('--dataset', default='coco', choices=['coco'])

    args = parser.parse_args()
    assert args.output_dir is not None or args.output_video is not None, (
        'One of --output-dir or --output-video must be specified.')

    with open(args.detectron_pickle, 'rb') as f:
        data = pickle.load(f)

    frames = sorted(data.keys(), key=lambda x: int(x))

    tracks = []
    track_id = 0
    images = []
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
            # features are of shape (num_segments, d, w, h). Average over w and
            # h, and convert to a list of length n with each element an array
            # of shape (d, ).
            mask_features = [
                x.mean(axis=(1, 2)) for x in image_data['features']
            ]

        detections = [
            Detection(box[:4], box[4], label, timestamp, image, mask,
                      feature) for box, mask, label, feature in zip(
                          boxes, masks, labels, mask_features)
            if box[4] > CONTINUE_TRACK_THRESHOLD  #  and label == 1  # person
        ]
        matched_tracks = match_detections(tracks, detections)
        # print('Timestamp: %s, Num matched tracks: %s' %
        #       (timestamp, len([x for x in matched_tracks if x is not None])))

        continued_tracks = []
        for detection, track in zip(detections, matched_tracks):
            if track is None:
                if detection.score > NEW_TRACK_THRESHOLD:
                    track = Track(track_id)
                    track_id += 1
                else:
                    continue

            track.add_detection(detection, timestamp)
            continued_tracks.append(track)

        continued_track_ids = set([x.id for x in continued_tracks])
        skipped_tracks = []
        for track in tracks:
            if track.id not in continued_track_ids and (
                    track.last_timestamp() - timestamp) < MAX_SKIP:
                skipped_tracks.append(track)

        tracks = continued_tracks + skipped_tracks

        new_image = visualize_detections(
            image, [
                track.detections[-1] for track in tracks
                if track.last_timestamp() == timestamp
            ],
            dataset=args.dataset)
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

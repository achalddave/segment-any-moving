"""Playground for tracking objects."""

import argparse
import os
import pickle

import cv2
import numpy as np
import PIL
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

import utils.vis as vis
from utils.colors import colormap
from utils.datasets import get_classes

# Detection confidence threshold for starting a new track
NEW_TRACK_THRESHOLD = 0.9
# Detection confidence threshold for adding a detection to existing track.
CONTINUE_TRACK_THRESHOLD = 0.5

# How many frames a track is allowed to miss detections in.
MAX_SKIP = 30

MATCH_SIMILARITY_THRESHOLD = 0.1


class Detection():
    def __init__(self, box, score, label, timestamp):
        self.box = box
        self.score = score
        self.label = label
        self.timestamp = timestamp
        self.track = None


class Track():
    def __init__(self, track_id):
        self.detections = []
        self.id = track_id

    def add_detection(self, detection, timestamp):
        detection.track = self
        self.detections.append(detection)

    def last_timestamp(self):
        if self.detections:
            return self.detections[-1].timestamp
        else:
            return None


def iou(box_first, box_second):
    """
    >>> iou([0, 0, 4, 4], [1, 1, 4, 4])
    0.5625
    """
    x_start0, y_start0, x_end0, y_end0 = box_first
    x_start1, y_start1, x_end1, y_end1 = box_second

    w_intersection = min(y_end0, y_end1) - max(y_start0, y_start1)
    h_intersection = min(x_end0, x_end1) - max(x_start0, x_start1)
    if w_intersection < 0 or h_intersection < 0:
        return 0
    intersection_area = w_intersection * h_intersection

    w_union = min(y_start0, y_start1) - max(y_end0, y_end1)
    h_union = min(x_start0, x_start1) - max(x_end0, x_end1)
    union_area = w_union * h_union

    return intersection_area / union_area


def track_similarity(track, detection):
    track_box = track.detections[-1].box
    detection_box = detection.box
    return iou(track_box, detection_box)


def match_detections(tracks, detections, threshold=0.5):
    matched_tracks = [None for _ in detections]
    left_indices = set(range(len(detections)))
    tracks = sorted(tracks, key=lambda t: t.last_timestamp(), reverse=True)
    for track in tracks:
        similarities = [np.float('-inf') for _ in detections]
        for i in left_indices:
            similarities[i] = track_similarity(track, detections[i])
        best_match = np.argmax(similarities)
        print('Best distance: %s' % max(similarities))
        if similarities[best_match] >= threshold:
            left_indices.remove(best_match)
            matched_tracks[best_match] = track
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
        image = vis.vis_bbox(image, (x0, y0, x1 - x0, y1 - y0), color, thick=3)

        label_str = '({track}) {label}: {score}'.format(
            track=detection.track.id,
            label=label_list[detection.label],
            score='{:0.2f}'.format(detection.score).lstrip('0'))
        image = vis.vis_class(image, (x0, y0 - 2), label_str)

    return image


def main():
    # Use first line of file docstring as description if it exists.
    argparse
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
    for timestamp, image_name in enumerate(tqdm(frames[:150])):
        image = cv2.imread(
            os.path.join(args.images_dir, image_name + args.extension))
        image = image[:, :, ::-1]  # BGR -> RGB
        image_data = data[image_name]
        boxes, _, _, labels = vis.convert_from_cls_format(
            image_data['boxes'], image_data['segmentations'],
            image_data['keypoints'])

        detections = [
            Detection(box[:4], box[4], label, timestamp)
            for box, label in zip(boxes, labels)
            if box[4] > CONTINUE_TRACK_THRESHOLD and label == 1  # person
        ]
        matched_tracks = match_detections(
            tracks,
            detections,
            threshold=MATCH_SIMILARITY_THRESHOLD)
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

        new_image = PIL.Image.fromarray(new_image)
        if args.output_dir is not None:
            new_image.save(
                os.path.join(args.output_dir, image_name + '.png'))

    if args.output_video is not None:
        clip = ImageSequenceClip(images, fps=args.output_video_fps)
        clip.write_videofile(args.output_video)


if __name__ == "__main__":
    main()

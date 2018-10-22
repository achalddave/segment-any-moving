"""Visualize nearest neighbors for a few detected objects."""

import argparse
import logging
import os
import pickle
import random
from pathlib import Path

import cv2
import PIL
import pycocotools.mask as mask_util
from scipy.spatial.distance import cosine
from joblib import Parallel, delayed
from tqdm import tqdm

import utils.vis as vis
from utils.distance import chi_square_distance, NeighborsQueue
from tracker.track import Detection

MIN_QUERY_SCORE = 0.9
MIN_NEIGHBOR_SCORE = 0.7

def detections_from_detectron_data(detectron_data, image, timestamp):
    boxes, masks, _, labels = vis.convert_from_cls_format(
        detectron_data['boxes'], detectron_data['segmentations'],
        detectron_data['keypoints'])

    mask_features = [None for mask in masks]
    if 'features' in detectron_data:
        # features are of shape (num_segments, d)
        mask_features = list(detectron_data['features'])
    assert len(boxes) == len(masks) == len(labels)
    detections = [
        Detection(box[:4], box[4], label, timestamp, image, mask,
                  feature) for box, mask, label, feature in zip(
                      boxes, masks, labels, mask_features)
    ]
    return detections


def save_image(image_np, output_path):
    PIL.Image.fromarray(image_np).save(output_path)


def compute_histogram_helper(detection):
    return detection.compute_histogram()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detectron-dir', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output-neighbors-dir', required=True)
    parser.add_argument('--extension', default='.png')
    parser.add_argument('--dataset', default='coco', choices=['coco'])
    parser.add_argument(
        '--filename-format',
        choices=['frame', 'sequence_frame', 'fbms'],
        default='frame',
        help=('Specifies how to get frame number from the filename. '
              '"frame": the filename is the frame number, '
              '"sequence_frame": the frame number is separated by an '
              'underscore'
              '"fbms": assume fbms style frame numbers'))
    parser.add_argument('--num-queries', default=5, type=int,
                        help='How many queries to visualize neighbors for')
    parser.add_argument('--num-neighbors', default=50, type=int,
                        help='How many neighbors to visualize')
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()
    random.seed(args.seed)

    os.makedirs(args.output_neighbors_dir, exist_ok=True)

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                        datefmt='%H:%M:%S')

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

    logging.info('Loading detections and images')
    detections_by_frame = {}
    for frame in tqdm(frames):
        timestamp = get_framenumber(frame)
        image = cv2.imread(
            os.path.join(args.images_dir, frame + args.extension))
        image = image[:, :, ::-1]  # BGR -> RGB
        detections = detections_from_detectron_data(
            data[frame], image, timestamp)
        detections_by_frame[frame] = [
            x for x in detections if x.score >= MIN_NEIGHBOR_SCORE
        ]

    logging.info('Precomputing histograms')
    flattened_detections = [x for y in detections_by_frame.values() for x in y]
    Parallel(n_jobs=8)(delayed(lambda x: x.compute_histogram())(d)
                       for d in tqdm(flattened_detections))

    logging.info('Finding neighbors')
    mask_color = [205, 168, 255]
    query_format = """
    <div class='query-block'>
        <div class='query'>
            <img src='{query_path}' />
        </div>
        <div class='neighbors'>
            {neighbors}
        </div>
    </div>
    """
    neighbor_format = "<img class='neighbor' src='{neighbor_path}' />"
    output_html = ''

    sampled_frames = random.sample([
        frame for frame in frames
        if any(d.score >= MIN_QUERY_SCORE for d in detections_by_frame[frame])
    ], args.num_queries)
    for query_name in tqdm(sampled_frames):
        query_detections = [
            x for x in detections_by_frame[query_name]
            if x.score >= MIN_QUERY_SCORE
        ]
        query_detections = [x for x in query_detections if x.label == 1]
        query_detection = random.choice(query_detections)

        # Save query image
        query_image_mask = vis.vis_mask(
            query_detection.image,
            query_detection.decoded_mask(),
            color=mask_color,
            alpha=0.0,
            border_thick=5)
        query_path = 'query-%s.png' % query_name
        save_image(query_image_mask,
                   os.path.join(args.output_neighbors_dir, query_path))

        neighbors = NeighborsQueue(maxsize=args.num_neighbors)

        Parallel(n_jobs=8)(delayed(lambda x: x.compute_histogram())(d)
                           for d in tqdm(flattened_detections))

        query_histogram = query_detection.compute_histogram()[0]
        dist_fn = delayed(chi_square_distance)
        distances = Parallel(n_jobs=8)(dist_fn(query_histogram,
                                               d.compute_histogram()[0])
                                       for d in tqdm(flattened_detections))
        for detection, distance in zip(flattened_detections, distances):
            neighbors.put(detection, distance)

        # List of (detection, distance) tuples.
        neighbors_list = [neighbors.get() for _ in range(args.num_neighbors)]
        # neighbors are returned in furthest-to-closest order; reverse that.
        neighbors_list = neighbors_list[::-1]

        neighbors_html = ''
        for neighbor_index, neighbor in enumerate(neighbors_list):
            neighbor_detection, distance = neighbor
            neighbor_mask = vis.vis_mask(
                neighbor_detection.image,
                neighbor_detection.decoded_mask(),
                color=mask_color,
                alpha=0.0,
                border_thick=5)
            neighbor_file = ('query-%s-neighbor-%s-frame-%s-distance-%.4f.png'
                             % (query_name, neighbor_index,
                                neighbor_detection.timestamp, 100 * distance))
            neighbors_html += neighbor_format.format(
                neighbor_path=neighbor_file)
            save_image(neighbor_mask,
                       os.path.join(args.output_neighbors_dir, neighbor_file))
        output_html += query_format.format(
            query_path=query_path, neighbors=neighbors_html)
    output_html = """
    <html>
    <style type='text/css'>
    img {{
        height: 200px;
    }}
    .neighbors {{
        height: 200px;
        overflow-x: scroll;
        white-space: nowrap;
    }}
    </style>
    <body>
    {}
    </body>
    </html>
    """.format(output_html)
    with open(os.path.join(args.output_neighbors_dir, 'neighbors.html'),
              'w') as f:
        f.write(output_html)


if __name__ == "__main__":
    main()

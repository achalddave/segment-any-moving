import argparse
import logging
import pathlib
import pickle
from pathlib import Path

import pycocotools.mask as mask_util
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from scipy.misc import imsave

import utils.vis as vis
from utils.colors import colormap
from utils.fbms.utils import (FbmsGroundtruth, get_tracks_text,
                              masks_to_tracks, get_framenumber)
from utils.log import setup_logging


def process_sequences(fbms_dir,
                      detectron_dir,
                      output_dir,
                      save_images,
                      detectron_threshold,
                      iou_threshold):
    assert fbms_dir.exists()
    assert detectron_dir.exists()

    output_dir.mkdir(exist_ok=True)

    sequence_paths = list(fbms_dir.iterdir())
    sequence_names = [x.name for x in sequence_paths]

    output_paths = []
    for sequence, sequence_path in zip(tqdm(sequence_names), sequence_paths):
        groundtruth_path = sequence_path / 'GroundTruth'
        assert groundtruth_path.exists(), (
            'Path %s does not exists' % groundtruth_path)
        groundtruth = FbmsGroundtruth(groundtruth_path)
        frame_number_to_labels = groundtruth.frame_labels()
        detectron_paths = (detectron_dir / sequence).glob('*.pickle')
        detectron_paths = sorted(
            detectron_paths, key=lambda x: get_framenumber(x.stem))

        final_masks = {}
        for frame_number, frame_labels in frame_number_to_labels.items():
            groundtruth_masks = []
            for color, region_id in groundtruth.color_to_region.items():
                if region_id == 0:
                    # ppms have full white (255 * 256**2 + 255 * 256 + 255)
                    # as background, pgms have 0 as background.
                    assert color == 16777215 or color == 0
                    continue  # Ignore background
                groundtruth_masks.append(frame_labels == region_id)

            # Last frame may not have predictions, use second to last frame.
            if frame_number == len(detectron_paths):
                logging.info(
                    ("No predictions found for frame %s in sequence %s, "
                     "using previous frame (%s) instead.") %
                    (frame_number, sequence, frame_number - 1))
                frame_number -= 1
            detectron_path = detectron_paths[frame_number]
            assert detectron_path.exists(), (
                '%s does not exist.' % detectron_path)

            with open(detectron_path, 'rb') as f:
                data = pickle.load(f)

            predicted_boxes, predicted_masks, _, _ = (
                vis.convert_from_cls_format(data['boxes'],
                                            data['segmentations'],
                                            data['keypoints']))
            if predicted_boxes is None:
                final_masks[frame_number] = np.zeros(
                    groundtruth_masks[0].shape, dtype=np.uint8)
                continue

            scores = predicted_boxes[:, -1]
            if np.all(scores <= detectron_threshold):
                logging.info('No masks above threshold (%s) Using most '
                             'confident mask only.' % detectron_threshold)
                predicted_masks = [predicted_masks[np.argmax(scores)]]
            else:
                predicted_masks = [
                    m for i, m in enumerate(predicted_masks)
                    if scores[i] > detectron_threshold
                ]
            predicted_masks = mask_util.decode(predicted_masks)
            predicted_masks = [
                predicted_masks[:, :, i]
                for i in range(predicted_masks.shape[2])
            ]

            mask_distance = np.zeros(
                (len(groundtruth_masks), len(predicted_masks)))
            mask_iou = mask_util.iou(
                [mask_util.encode(p) for p in predicted_masks],
                [mask_util.encode(np.asfortranarray(g.astype('uint8')))
                 for g in groundtruth_masks],
                pyiscrowd=np.zeros(len(groundtruth_masks)))

            assert isinstance(mask_iou, np.ndarray), (
                'Unknown type of mask_iou (%s) for sequence %s, frame %s' %
                (type(mask_iou), sequence, frame_number))

            filtered_prediction_indices = np.where(
                np.any(mask_iou >= iou_threshold, axis=1))[0]
            mask_iou = mask_iou[filtered_prediction_indices]
            filtered_predictions = [
                predicted_masks[x] for x in filtered_prediction_indices
            ]
            mask_distance = 1 - mask_iou

            # Array of length num_matches, containing tuples of
            # (predicted_mask_index, groundtruth_mask_index)
            assignments = list(zip(*linear_sum_assignment(mask_distance)))
            final_mask = np.zeros(groundtruth_masks[0].shape, dtype=np.uint8)
            if False:
                from matplotlib import pyplot as plt
                plt.close()
                _, ax = plt.subplots(len(assignments), 2)
                plt.suptitle('Frame %s' % frame_number)

            for predicted_mask_index, groundtruth_id in assignments:
                predicted_mask = filtered_predictions[predicted_mask_index]
                final_mask[predicted_mask != 0] = groundtruth_id + 1
                if False:
                    ax[groundtruth_id, 0].imshow(groundtruth_masks[groundtruth_id])
                    ax[groundtruth_id, 0].title.set_text('Groundtruth')
                    ax[groundtruth_id, 1].imshow(predicted_mask)
                    ax[groundtruth_id, 1].title.set_text(
                        'Predicted; iou: %.4f' %
                        (1 - mask_distance[predicted_mask_index, groundtruth_id]))
            if False:
                plt.show()
            final_masks[frame_number] = final_mask

        tracks = masks_to_tracks(final_masks)
        tracks_str = get_tracks_text(tracks, groundtruth.num_frames)
        output_file = output_dir / (sequence + '.dat')
        output_paths.append(output_file)
        with open(output_file, 'w') as f:
            f.write(tracks_str)

        if save_images:
            output_images = output_dir / (sequence + '-images')
            output_images.mkdir(exist_ok=True)
            colors = colormap()  # list(range(0, 251, 25))
            full_output = None
            for frame_number, frame_labels in frame_number_to_labels.items():
                groundtruth_output = np.zeros((frame_labels.shape[0],
                                               frame_labels.shape[1], 3))
                predictions_output = np.zeros((frame_labels.shape[0],
                                               frame_labels.shape[1], 3))
                for color, region_id in groundtruth.color_to_region.items():
                    if region_id == 0:
                        color = (255, 255, 255)
                    else:
                        color = colors[region_id - 1]
                    groundtruth_output[frame_labels == region_id] = color
                    predictions_output[final_masks[frame_number] ==
                                       region_id] = (color)
                concatenated = np.hstack((groundtruth_output,
                                          predictions_output))
                if full_output is None:
                    full_output = concatenated
                else:
                    full_output = np.vstack((full_output, concatenated))
                # imsave(output_images / ('groundtruth-%s.jpg' % frame_number),
                #        groundtruth_output)
                # imsave(output_images / ('predictions-%s.jpg' % frame_number),
                #        predictions_output)
            imsave(output_images / 'final.jpg', full_output)

    with open(output_dir / 'all_tracks.txt', 'w') as f:
        for output_path in output_paths:
            f.write(str(output_path.resolve()) + '\n')

    with open(output_dir / 'all_shots.txt', 'w') as f:
        f.write(str(len(sequence_paths)) + '\n')
        for sequence, sequence_path in zip(sequence_names, sequence_paths):
            groundtruth_path = sequence_path / 'GroundTruth' / (
                sequence + 'Def.dat')
            f.write(str(groundtruth_path.resolve()) + '\n')


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fbms-root', required=True)
    parser.add_argument('--detectron-outputs',
                        required=True,
                        help="""
                        Directory containing detectron outputs. Assumed to
                        contain TrainingSet/ and TestSet/ directories,
                        structured like
                        TrainingSet/<sequence>/<sequence>_<frame>.pickle, e.g.
                        TrainingSet/bear01/bear01_0000.pickle.""")
    parser.add_argument('--output-dir', required=True)
    parser.add_argument(
        '--set', choices=['train', 'test', 'all'], default='all')
    parser.add_argument(
        '--save-images', action='store_true')
    parser.add_argument(
        '--detectron-threshold', type=float, default=0.7)
    parser.add_argument('--iou-threshold', type=float, default=0.0)
    args = parser.parse_args()

    output = pathlib.Path(args.output_dir)
    output.mkdir(exist_ok=True)

    with open(__file__, 'r') as f:
        _source = f.read()

    logging_path = str(output / (Path(__file__).name + '.log'))
    setup_logging(str(logging_path))
    logging.info('Args: %s\n' % vars(args))

    file_logger = logging.getLogger(logging_path)

    fbms_root = pathlib.Path(args.fbms_root)
    assert fbms_root.exists()

    detectron_root = pathlib.Path(args.detectron_outputs)
    assert detectron_root.exists()

    use_train = args.set in ('train', 'all')
    use_test = args.set in ('test', 'all')

    if use_train:
        train_detectron = detectron_root / 'TrainingSet'
        # If we only want to evaluate on train, and the "TrainingSet" directory
        # doesn't exist, then treat the root directory as the training
        # directory.
        if (not use_test and not train_detectron.exists()
                and detectron_root.exists()):
            train_detectron = detectron_root
        process_sequences(fbms_root / 'TrainingSet',
                          train_detectron,
                          output / 'TrainingSet',
                          args.save_images,
                          args.detectron_threshold,
                          args.iou_threshold)

    if use_test:
        test_detectron = detectron_root / 'TestSet'
        # If we only want to evaluate on train, and the "TestSet" directory
        # doesn't exist, then treat the root directory as the test
        # directory.
        if (not use_train and not test_detectron.exists()
                and detectron_root.exists()):
            test_detectron = detectron_root
        process_sequences(fbms_root / 'TestSet',
                          test_detectron,
                          output / 'TestSet',
                          args.save_images,
                          args.detectron_threshold,
                          args.iou_threshold)

    file_logger.info('Source:')
    file_logger.info('=======')
    file_logger.info(_source)


if __name__ == "__main__":
    main()

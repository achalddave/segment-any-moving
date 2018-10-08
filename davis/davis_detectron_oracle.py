import argparse
import pathlib
import pickle
import yaml
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from scipy.optimize import linear_sum_assignment

import utils.vis as vis


def get_unique_objects(groundtruth):
    """Get unique object ids from segmentation mask

    Adapted from DAVIS evaluation code.
    """
    ids = sorted(np.unique(groundtruth))
    if ids[-1] == 255:  # Remove unknown-label
        ids = ids[:-1]
    if ids[0] == 0:  # Remove background
        ids = ids[1:]
    return ids


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--davis-data-root',
        required=True,
        help='Directory containing a subdirectory for each sequence')
    parser.add_argument(
        '--davis-eval-root',
        required=True,
        help='DAVIS evaluation code root directory.')
    parser.add_argument(
        '--detectron-root',
        required=True,
        help=('Contains subdirectory for each sequence, containing pickle '
              'files of detectron outputs for each frame.'))
    parser.add_argument(
        '--set', choices=['train', 'val'], default='val')
    parser.add_argument('--output-dir', required=True)

    args = parser.parse_args()

    davis_eval_root = pathlib.Path(args.davis_eval_root)
    davis_root = pathlib.Path(args.davis_data_root)
    detectron_root = pathlib.Path(args.detectron_root)
    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(exist_ok=True)

    db_info_path = davis_eval_root / 'data' / 'db_info.yaml'
    assert db_info_path.exists(), (
        'DB info file (%s) does not exist' % db_info_path)
    with open(db_info_path, 'r') as f:
        davis_info = yaml.load(f)

    palette_path = Path(__file__).parent / 'palette.txt'
    assert palette_path.exists(), (
        'DAVIS palette file (%s) does not exist' % palette_path)
    palette = np.loadtxt(palette_path, dtype=np.uint8).reshape(-1, 3)

    for sequence_info in davis_info['sequences']:
        if sequence_info['set'] != args.set:
            continue
        sequence = sequence_info['name']
        print(sequence)
        output_sequence = output_root / sequence
        output_sequence.mkdir(exist_ok=True)
        detectron_sequence = detectron_root / sequence
        davis_sequence = davis_root / sequence
        assert detectron_sequence.exists(), (
            'Detectron path %s does not exist' % detectron_sequence)
        assert davis_sequence.exists(), (
            'DAVIS path %s does not exist' % davis_sequence)
        detectron_frames = sorted(
            detectron_sequence.glob('*.pickle'), key=lambda x: int(x.stem))
        davis_frames = sorted(
            davis_sequence.glob('*.png'), key=lambda x: int(x.stem))
        num_frames = sequence_info['num_frames']

        for frame, detectron_path, davis_path in zip(
                range(num_frames), detectron_frames, davis_frames):
            output_frame = output_sequence / ('%05d.png' % frame)
            groundtruth = np.array(Image.open(davis_path))
            object_ids = get_unique_objects(groundtruth)
            groundtruth_masks = [groundtruth == i for i in object_ids]
            with open(detectron_path, 'rb') as f:
                data = pickle.load(f)
            predicted_boxes, predicted_masks, _, _ = (
                vis.convert_from_cls_format(
                    data['boxes'], data['segmentations'], data['keypoints']))
            # Can threshold scores if necessary
            # scores = predicted_boxes[:, -1]
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

            mask_distance = 1 - mask_iou

            # Array of length num_matches, containing tuples of
            # (predicted_mask_index, groundtruth_mask_index)
            assignments = list(zip(*linear_sum_assignment(mask_distance)))
            final_mask = np.zeros(groundtruth_masks[0].shape, dtype=np.uint8)

            for predicted_mask_index, groundtruth_id in assignments:
                predicted_mask = predicted_masks[predicted_mask_index]
                final_mask[predicted_mask != 0] = object_ids[groundtruth_id]

            output = Image.fromarray(final_mask)
            output.putpalette(palette.ravel())
            output.save(output_frame, format='png')

        assert len(detectron_frames) == len(davis_frames) == num_frames


if __name__ == "__main__":
    main()

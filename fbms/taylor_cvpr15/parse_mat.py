import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

import utils.fbms.utils as fbms_utils


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--mat-dir',
        type=Path,
        required=True,
        help=('Contains subdirectories TrainingSet and TestSet, each of which '
              'contain files of the form <seq>_obj.mat and '
              '<seq>_obj_reverse.mat. Use rename_and_split.py to get the '
              'directory in this format, first.'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fbms-dir', type=Path, required=True)
    parser.add_argument(
        '--use-reverse',
        action='store_true',
        help='Whether to use the non-causal output.')

    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)
    output_dat_root = args.output_dir / 'dat-predictions'
    output_dat_root.mkdir(exist_ok=True)
    output_np_root = args.output_dir / 'numpy-predictions'
    output_np_root.mkdir(exist_ok=True)

    for split in ['TrainingSet', 'TestSet']:
        output_dat_split = output_dat_root / split
        output_dat_split.mkdir(exist_ok=True)
        output_np_split = output_np_root / split
        output_np_split.mkdir(exist_ok=True)

        if args.use_reverse:
            sequence_files = (args.mat_dir / split).glob('*_obj_reverse.mat')
        else:
            sequence_files = (args.mat_dir / split).glob('*_obj.mat')

        all_shot_paths = []
        all_track_paths = []
        for sequence_mat in tqdm(list(sequence_files)):
            sequence_name = sequence_mat.stem.split('_')[0]
            # (height, width, num_frames)
            segmentation = loadmat(sequence_mat)['obj']
            # (num_frames, height, width)
            segmentation = segmentation.transpose((2, 0, 1))
            output_np = output_np_split / (sequence_name + '.npy')
            assert not output_np.exists(), '%s already exists' % output_np
            np.save(output_np, segmentation)

            output_dat = output_dat_split / (sequence_name + '.dat')
            groundtruth_dir = (
                args.fbms_dir / split / sequence_name / 'GroundTruth')
            groundtruth = fbms_utils.FbmsGroundtruth(groundtruth_dir)
            # It seems that the FBMS evaluation code assumes that the number
            # of regions is equal to the max of the region ids + 1, while the
            # output in these .mat files contains non-contiguous region ids.
            # Here, we make them contiguous.
            ids = set()
            for frame in groundtruth.frame_label_paths.keys():
                ids.update(set(np.unique(segmentation[frame])))
            masks = {}
            for frame in groundtruth.frame_label_paths.keys():
                masks[frame] = np.zeros_like(segmentation[frame])
                for i, region_id in enumerate(sorted(ids)):
                    masks[frame][segmentation[frame] == region_id] = i

            for frame in masks:
                print(sequence_name, np.unique(masks[frame]))
            fbms_tracks = fbms_utils.masks_to_tracks(masks)
            fbms_tracks_str = fbms_utils.get_tracks_text(
                fbms_tracks, groundtruth.num_frames, verbose=False)
            with open(output_dat, 'w') as f:
                f.write(fbms_tracks_str)
            all_shot_paths.append(
                groundtruth_dir / (sequence_name + 'Def.dat'))
            all_track_paths.append(output_dat)

        with open(output_dat_split / 'all_shots.txt', 'w') as f:
            f.write(str(len(all_shot_paths)) + '\n')
            f.write('\n'.join(str(x.resolve()) for x in all_shot_paths))

        with open(output_dat_split / 'all_tracks.txt', 'w') as f:
            f.write('\n'.join(str(x.resolve()) for x in all_track_paths))


if __name__ == "__main__":
    main()

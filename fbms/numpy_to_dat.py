import argparse
import logging
from pathlib import Path

import numpy as np
from script_utils.common import common_setup
from tqdm import tqdm

from utils.fbms import utils as fbms_utils


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--numpy-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--fbms-groundtruth', type=Path, required=True)
    parser.add_argument('--np-extensions', default=['.npy', '.npz'])

    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)

    inputs = []
    for ext in args.np_extensions:
        inputs.extend(args.numpy_dir.glob('*' + ext))

    all_shot_paths = []
    all_track_paths = []
    for mask_file in tqdm(inputs):
        sequence = mask_file.stem
        output_file = args.output_dir / (sequence + '.dat')
        groundtruth_dir = args.fbms_groundtruth / sequence / 'GroundTruth'
        assert groundtruth_dir.exists(), (
            f"Couldn't find groundtruth at {groundtruth_dir}")
        groundtruth = fbms_utils.FbmsGroundtruth(groundtruth_dir)
        all_shot_paths.append(groundtruth_dir / (sequence + 'Def.dat'))
        all_track_paths.append(output_file)
        if output_file.exists():
            logging.info(f'Output {output_file} exists, skipping.')
            continue

        mask = np.load(mask_file)
        if isinstance(mask, np.lib.npyio.NpzFile):
            # Segmentation saved with savez_compressed; ensure there is only
            # one item in the dict and retrieve it.
            keys = list(mask.keys())
            assert len(keys) == 1, (
                'Numpy file (%s) contained dict with multiple items, not sure '
                'which one to load.' % mask_file)
            mask = mask[keys[0]]
        mask = {t: mask[t] for t in groundtruth.frame_label_paths.keys()}
        fbms_tracks = fbms_utils.masks_to_tracks(mask)
        fbms_tracks_str = fbms_utils.get_tracks_text(fbms_tracks,
                                                     groundtruth.num_frames)
        with open(output_file, 'w') as f:
            f.write(fbms_tracks_str)

    with open(args.output_dir / 'all_shots.txt', 'w') as f:
        f.write(str(len(all_shot_paths)) + '\n')
        f.write('\n'.join(str(x.resolve()) for x in all_shot_paths))

    with open(args.output_dir / 'all_tracks.txt', 'w') as f:
        f.write('\n'.join(str(x.resolve()) for x in all_track_paths))


if __name__ == "__main__":
    main()
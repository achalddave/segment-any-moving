"""Convert PNG masks to numpy arrays."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'input_dir',
        help=('Contains subdirectory for each sequence, each of which '
              'contains <frame>.png for each frame'),
        type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--duplicate-last-frame', action='store_true')

    args = parser.parse_args()

    if not args.duplicate_last_frame:
        print('NOTE: Not duplicating last frame. This could cause issues '
              'with evaluation.')
    args.output_dir.mkdir(exist_ok=True, parents=True)
    for sequence_dir in tqdm(list(args.input_dir.iterdir())):
        if not sequence_dir.is_dir():
            continue
        images = list(sequence_dir.iterdir())
        all_frames = [None] * len(images)
        for image_path in images:
            frame = int(image_path.stem)-1
            all_frames[frame] = np.array(Image.open(image_path))
        if args.duplicate_last_frame:
            all_frames.append(all_frames[-1])
        all_frames = np.stack(all_frames)
        np.save((args.output_dir / sequence_dir.name).with_suffix('.npy'),
                all_frames)


if __name__ == "__main__":
    main()

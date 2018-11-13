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
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--np-extension', default='.npy')
    parser.add_argument('--num-colors', default=100, type=int)

    args = parser.parse_args()

    colors = (np.random.rand(args.num_colors, 3) * 256).round()

    for mask_file in tqdm(list(args.input_dir.rglob('*' + args.np_extension))):
        all_frames_mask = np.load(mask_file)

        num_frames = all_frames_mask.shape[0]
        for f in range(num_frames):
            frame_mask = all_frames_mask[f]
            ids = sorted(np.unique(frame_mask))
            masks = [frame_mask == object_id for object_id in ids]
            # Sort masks by area
            ids_and_masks = sorted(zip(ids, masks), key=lambda x: x[1].sum())

            vis_mask = np.zeros(
                (frame_mask.shape[0], frame_mask.shape[1], 3),
                dtype=np.uint8)
            for mask_id, mask in ids_and_masks:
                color = colors[mask_id % args.num_colors]
                vis_mask[mask == 1] = color
            output_frame = (args.output_dir / mask_file.parent.relative_to(
                args.input_dir) / mask_file.stem / (str(f) + '.png'))
            output_frame.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(vis_mask).save(output_frame)


if __name__ == "__main__":
    main()

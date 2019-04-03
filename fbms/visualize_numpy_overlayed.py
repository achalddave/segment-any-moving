"""Visualize numpy predictions overlayed on original images."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted, ns
from PIL import Image
from tqdm import tqdm

from utils.misc import is_image_file
from utils.vis import vis_mask
from utils.colors import colormap


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--images-dir', required=True, type=Path)
    parser.add_argument('--np-extension', default='.npy')
    parser.add_argument('--output-fps', default=30, type=int)
    parser.add_argument('--output-images', action='store_true')
    parser.add_argument(
        '--background-id',
        required=True,
        help=('ID of background track in predictions. Can be an integer or '
              '"infer", in which case the background id is assumed to be the '
              'id of the track with the most pixels.'))

    args = parser.parse_args()

    colors = colormap()
    if args.background_id != 'infer':
        background_prediction_id = int(args.background_id)
    else:
        background_prediction_id = None

    for mask_path in tqdm(list(args.input_dir.rglob('*' + args.np_extension))):
        relative_dir = mask_path.relative_to(args.input_dir).with_suffix('')
        images_subdir = args.images_dir / relative_dir
        assert images_subdir.exists(), (
            'Could not find directory %s' % images_subdir)
        images = natsorted(
            [x for x in images_subdir.iterdir() if is_image_file(x.name)],
            alg=ns.PATH)

        all_frames_mask = np.load(mask_path)
        if args.np_extension == '.npz':
            # Segmentation saved with savez_compressed; ensure there is only
            # one item in the dict and retrieve it.
            keys = all_frames_mask.keys()
            assert len(keys) == 1, (
                'Numpy file (%s) contained dict with multiple items, not sure '
                'which one to load.' % mask_path)
            all_frames_mask = all_frames_mask[keys[0]]
        all_ids, id_counts = np.unique(all_frames_mask, return_counts=True)
        id_counts = dict(zip(all_ids, id_counts))
        sorted_ids = sorted(
            id_counts.keys(), key=lambda i: id_counts[i], reverse=True)
        if background_prediction_id is None:  # Infer background id
            current_bg = int(sorted_ids[0])
            print('Inferred background prediction id as %s for %s' %
                  (current_bg, relative_dir))
            sorted_ids = sorted_ids[1:]
        else:
            current_bg = background_prediction_id
            sorted_ids = [
                x for x in sorted_ids if x != current_bg
            ]

        # Map id to size index
        id_rankings = {
            region_id: index
            for index, region_id in enumerate(sorted_ids)
        }

        def visualize_frame(t):
            frame = int(t * args.output_fps)
            frame_mask = all_frames_mask[frame]
            image_path = images[frame]
            ids = sorted(np.unique(frame_mask))
            masks = [frame_mask == object_id for object_id in ids]

            # Sort masks by area
            ids_and_masks = sorted(zip(ids, masks), key=lambda x: x[1].sum())
            vis_image = cv2.imread(str(image_path))
            # vis_image = (vis_image.astype(np.float32) * 1.0).astype(np.uint8)
            for mask_id, mask in ids_and_masks:
                if isinstance(mask_id, float):
                    assert mask_id.is_integer()
                    mask_id = int(mask_id)
                if mask_id == current_bg:
                    continue
                color = colors[int(id_rankings[mask_id]) % len(colors)]
                vis_image = vis_mask(
                    vis_image,
                    mask.astype(np.uint8),
                    color,
                    alpha=0.5,
                    border_alpha=0.5,
                    border_color=[255, 255, 255],
                    border_thick=2)
            vis_image = vis_image[:, :, ::-1]  # BGR -> RGB
            if args.output_images:
                output_frame = args.output_dir / image_path.relative_to(
                    args.images_dir)
                output_frame.parent.mkdir(exist_ok=True, parents=True)
                Image.fromarray(vis_image).save(output_frame)
            return vis_image

        num_frames = all_frames_mask.shape[0]
        output_video = (args.output_dir / relative_dir).with_suffix('.mp4')
        output_video.parent.mkdir(exist_ok=True, parents=True)
        from moviepy.video.VideoClip import VideoClip
        clip = VideoClip(make_frame=visualize_frame)
        # Subtract a small epsilon; otherwise, moviepy can sometimes request
        # a frame at index num_frames.
        duration = num_frames / args.output_fps - 1e-10
        clip = clip.set_duration(duration).set_memoize(True)
        clip.write_videofile(
            str(output_video), fps=args.output_fps, verbose=False)


if __name__ == "__main__":
    main()

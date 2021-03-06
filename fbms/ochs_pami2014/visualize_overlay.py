"""Overlay dense predictions onto original images using our visualization."""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from natsort import natsorted, ns
from PIL import Image
from tqdm import tqdm

from utils.misc import is_image_file
from utils.vis import vis_mask
from utils.colors import colormap
from utils.log import setup_logging


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dense-dir',
        required=True,
        type=Path,
        help='Directory containing dense segmentation .ppm files.')
    parser.add_argument('--images-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--output-fps', default=30, type=int)
    parser.add_argument('--output-images', action='store_true')
    parser.add_argument(
        '--background-id',
        required=True,
        help=('ID of background track in predictions. Can be an integer or '
              '"infer", in which case the background id is assumed to be the '
              'id of the track with the most pixels.'))

    args = parser.parse_args()

    assert args.dense_dir.exists()
    assert args.images_dir.exists()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    setup_logging(args.output_dir / (Path(__file__).name + '.log'))
    logging.info('File path: %s', Path(__file__))
    logging.info('Args:\n%s', vars(args))

    colors = colormap()
    if args.background_id != 'infer':
        background_prediction_id = int(args.background_id)
    else:
        background_prediction_id = None

    dense_segmentations = natsorted(
        args.dense_dir.glob('*_dense.ppm'), alg=ns.PATH)
    images = natsorted(
        [x for x in args.images_dir.iterdir() if is_image_file(x.name)],
        alg=ns.PATH)
    assert len(images) == len(dense_segmentations)

    segmentation_frames = np.stack(
        np.array(Image.open(segmentation_ppm))
        for segmentation_ppm in dense_segmentations)
    if segmentation_frames.ndim == 4 and segmentation_frames.shape[-1] == 1:
        segmentation_frames = segmentation_frames[:, :, :, 0]
    elif segmentation_frames.ndim == 4 and segmentation_frames.shape[-1] == 3:
        segmentation_frames = segmentation_frames.astype(np.int32)
        segmentation_frames = (segmentation_frames[:, :, :, 2] +
                               256 * segmentation_frames[:, :, :, 1] +
                               (256**2) * segmentation_frames[:, :, :, 0])
        assert segmentation_frames.ndim == 3

    all_ids, id_counts = np.unique(segmentation_frames, return_counts=True)
    id_counts = dict(zip(all_ids, id_counts))
    sorted_ids = sorted(
        id_counts.keys(), key=lambda i: id_counts[i], reverse=True)
    if background_prediction_id is None:  # Infer background id
        background_prediction_id = int(sorted_ids[0])
        print('Inferred background prediction id as %s' %
              background_prediction_id)
        sorted_ids = sorted_ids[1:]
    else:
        sorted_ids = [
            x for x in sorted_ids if x != background_prediction_id
        ]
    # Map id to size index
    id_rankings = {
        region_id: index
        for index, region_id in enumerate(sorted_ids)
    }

    def visualize_frame(t):
        frame = int(t * args.output_fps)
        frame_mask = segmentation_frames[frame]
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
            if mask_id == background_prediction_id:
                continue
            color = colors[int(id_rankings[mask_id]) % len(colors)]
            vis_image = vis_mask(
                vis_image,
                mask.astype(np.uint8),
                color,
                alpha=0.5,
                border_alpha=0.5,
                border_color=[255, 255, 255],
                border_thick=1)
        vis_image = vis_image[:, :, ::-1]  # BGR -> RGB
        if args.output_images:
            output_frame = args.output_dir / image_path.name
            output_frame.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(vis_image).save(output_frame)
        return vis_image

    num_frames = segmentation_frames.shape[0]
    output_video = args.output_dir / 'video.mp4'
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

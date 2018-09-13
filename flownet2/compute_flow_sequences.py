"""Compute LiteFlowNet flow on videos."""

import argparse
import contextlib
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from liteflownet.convert_flo_png import convert_flo
from utils.log import setup_logging


@contextlib.contextmanager
def gpu_from_queue(gpu_queue):
    gpu = gpu_queue.get()
    yield gpu
    gpu_queue.put(gpu)


def compute_sequence_flow(image_paths, output_dir, prototxt, caffe_model,
                          flownet_root, tmp_prefix, gpu_queue, logger_name,
                          convert_png):
    times = {}
    times['start'] = time.time()
    file_logger = logging.getLogger(logger_name)
    dimensions = None
    for image_path in image_paths:
        image = np.array(Image.open(image_path))
        if dimensions is None:
            dimensions = image.shape
        else:
            assert dimensions == image.shape, (
                'Image sizes in sequence do not match (%s: %s, vs %s: %s)' %
                (image_paths[0], dimensions, image_path, image.shape))

    output_dir.mkdir(exist_ok=True, parents=True)

    # Final command:
    # run-flownet-many.py /path/to/$net/$net_weights.caffemodel[.h5] \
    #                     /path/to/$net/$net_deploy.prototxt.template \
    #                      list.txt
    #
    # (where list.txt contains lines of the form "x.png y.png z.flo")

    # List of tuples: (frame1_path, frame2_path, output_flo)
    flo_input_outputs = []

    # List of tuples: (flo_input, output_png, output_metadata).
    # If convert_png=False, this list is empty.
    png_input_outputs = []

    for frame1, frame2 in zip(image_paths[:-1], image_paths[1:]):
        flo = output_dir / (frame1.stem + '.flo')

        if convert_png:
            png = flo.with_suffix('.png')
            metadata = flo.with_name(flo.stem + '_magnitude_minmax.txt')
            # Already computed everything, skip this frame pair.
            if png.exists() and metadata.exists():
                continue
            png_input_outputs.append((flo, png, metadata))

        if not flo.exists():
            flo_input_outputs.append((frame1, frame2, flo))

    os.environ['CAFFE_PATH'] = str(flownet_root)
    os.environ['PYTHONPATH'] = '%s:%s' % (flownet_root / 'python',
                                          os.environ['PYTHONPATH'])
    os.environ['LD_LIBRARY_PATH'] = '%s:%s' % (flownet_root / 'build' / 'lib',
                                               os.environ['LD_LIBRARY_PATH'])

    if flo_input_outputs:
        times['gpu_wait_start'] = time.time()
        with NamedTemporaryFile('w', prefix=tmp_prefix) as input_list_f, \
                gpu_from_queue(gpu_queue) as gpu:
            times['gpu_wait_end'] = time.time()
            for frame1, frame2, output_flo in flo_input_outputs:
                input_list_f.write('%s %s %s\n' % (frame1, frame2, output_flo))
            input_list_f.flush()

            command = [
                'python',
                str(flownet_root / 'scripts' / 'run-flownet-many.py'),
                str(caffe_model), str(prototxt), input_list_f.name,
                '--gpu', str(gpu)
            ]

            file_logger.info('Executing %s' % ' '.join(command))
            try:
                subprocess.check_output(command, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                logging.fatal('Failed command.\nException: %s\nOutput %s',
                              e.returncode, e.output)
                raise
    else:
        times['gpu_wait_start'] = times['gpu_wait_end'] = 0

    # png_input_outputs will be empty if convert_png=False or if we have
    # already converted all the flows.
    for flo_path, png_path, metadata_path in png_input_outputs:
        try:
            convert_flo(flo_path, png_path, metadata_path)
        except Exception as e:
            logging.error('ERROR converting flo path: %s' % flo_path)
            raise e
        flo_path.unlink()

    time_taken = (time.time() - times['start']) - (
        times['gpu_wait_end'] - times['gpu_wait_start'])
    if not flo_input_outputs and not png_input_outputs:
        file_logger.info(
            'Output dir %s was already processed, skipping. Time taken: %s' %
            (output_dir, time_taken))
    else:
        file_logger.info(
            'Processed %s. Time taken: %s' % (output_dir, time_taken))


def compute_sequence_flow_gpu_helper(kwargs):
    return compute_sequence_flow(**kwargs)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing a subdir for every sequence.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--flownet2-dir', required=True)
    parser.add_argument(
        '--recursive',
        action='store_true',
        help="""Search recursively in input-dir for sequences. Any directory
                containing a file with extension specified by --extension is
                treated as a sequence directory. NOTE: Does not support
                symlinked directories.""")
    parser.add_argument('--extension', default='.png')
    parser.add_argument(
        '--cnn-model',
        default='kitti',
        choices=[
            'kitti', 'sintel', 'flyingthings'
        ])
    parser.add_argument(
        '--convert-to-angle-magnitude-png',
        help=('Convert flo files to angle/magnitude PNGs, and do not keep '
              '.flo files around.'),
        action='store_true')
    parser.add_argument('--gpus', default=[0, 1, 2, 3], nargs='*', type=int)
    parser.add_argument(
        '--num-workers',
        default=-1,
        type=int,
        help=('Number of workers. By default, set to the number of GPUs. '
              'Having more workers than GPUs allows some workers to process '
              'CPU operations, like loading input/output lists, checking '
              'image dimensions, and converting .flo to .png while other '
              'workers use the GPU.'))

    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    file_name = Path(__file__).stem
    logging_path = str(
        output_root /
        (file_name + '.py.%s.log' % datetime.now().strftime('%b%d-%H-%M-%S')))
    setup_logging(logging_path)
    logging.info('Args:\n%s', vars(args))

    if args.extension[0] != '.':
        args.extension = '.' + args.extension

    flownet_root = Path(args.flownet2_dir)

    cnn_model = args.cnn_model
    models_dir = flownet_root / 'models'
    model_info = {
        'kitti': {
            'weights':
                'FlowNet2-KITTI/FlowNet2-KITTI_weights.caffemodel.h5',
            'prototxt':
                'FlowNet2-KITTI/FlowNet2-KITTI_deploy.prototxt.template'
        },
        'sintel': {
            'weights':
                'FlowNet2-Sintel/FlowNet2-CSS-Sintel_weights.caffemodel.h5',
            'prototxt':
                'FlowNet2-Sintel/FlowNet2-CSS-Sintel_deploy.prototxt.template'
        },
        'flyingthings': {
            'weights': 'FlowNet2-Sintel/FlowNet2_weights.caffemodel.h5',
            'prototxt': 'FlowNet2-Sintel/FlowNet2_deploy.prototxt.template'
        }
    }

    weights = models_dir / model_info[cnn_model]['weights']
    prototxt = models_dir / model_info[cnn_model]['prototxt']

    assert weights.exists(), 'Caffe model file does not exist at %s' % weights
    assert prototxt.exists(), 'Caffe prototxt does not exist at %s' % prototxt

    if args.recursive:
        sequences = sorted(
            set(x.parent for x in input_root.rglob('*' + args.extension)))
    else:
        sequences = sorted(input_root.iterdir())

    import multiprocessing as mp
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    if args.num_workers == -1:
        args.num_workers = len(args.gpus)
    pool = mp.Pool(args.num_workers)
    for gpu in args.gpus:
        gpu_queue.put(gpu)

    tasks = []
    for sequence_path in sequences:
        output_dir = output_root / (sequence_path.relative_to(input_root))
        image_paths = natsorted(
            list(sequence_path.glob('*' + args.extension)),
            key=lambda x: x.stem)
        tasks.append({
            'gpu_queue': gpu_queue,
            'image_paths': image_paths,
            'output_dir': output_dir,
            'tmp_prefix': file_name,
            'prototxt': prototxt,
            'caffe_model': weights,
            'flownet_root': flownet_root,
            'logger_name': logging_path,
            'convert_png': args.convert_to_angle_magnitude_png
        })

    list(
        tqdm(
            pool.imap_unordered(compute_sequence_flow_gpu_helper, tasks),
            total=len(tasks)))


if __name__ == "__main__":
    main()

"""Compute LiteFlowNet flow on videos."""

import argparse
import logging
import subprocess
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from liteflownet.convert_flo_png import convert_flo
from utils.log import setup_logging


def compute_sequence_flow(image_paths, output_dir, prototxt, caffe_model,
                          flownet_root, tmp_prefix, gpu, logger_name,
                          convert_png):
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

    input_outputs = []  # list of (frame1_path, frame2_path, output_path)
    for frame1, frame2 in zip(image_paths[:-1], image_paths[1:]):
        input_outputs.append((frame1, frame2,
                              output_dir / (frame1.stem + '.flo')))

    os.environ['CAFFE_PATH'] = str(flownet_root)
    os.environ['PYTHONPATH'] = '%s:%s' % (flownet_root / 'python',
                                          os.environ['PYTHONPATH'])
    os.environ['LD_LIBRARY_PATH'] = '%s:%s' % (flownet_root / 'build' / 'lib',
                                               os.environ['LD_LIBRARY_PATH'])

    with NamedTemporaryFile('w', prefix=tmp_prefix) as input_list_f:
        for frame1, frame2, output_file in input_outputs:
            input_list_f.write('%s %s %s\n' % (frame1, frame2, output_file))
        input_list_f.flush()

        command = [
            'python', str(flownet_root / 'scripts' / 'run-flownet-many.py'),
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

        for input_path, _, output_path in input_outputs:
            if convert_png:
                output_flo_png = (
                    output_dir / input_path.stem).with_suffix('.png')
                output_flo_metadata = (
                    output_dir / (input_path.stem + '_magnitude_minmax.txt'))
                convert_flo(output_path, output_flo_png, output_flo_metadata)
                output_path.unlink()


def compute_sequence_flow_gpu_helper(kwargs):
    gpu_queue = kwargs['gpu_queue']
    gpu = gpu_queue.get()
    try:
        del kwargs['gpu_queue']
        kwargs['gpu'] = gpu
        return compute_sequence_flow(**kwargs)
    finally:
        gpu_queue.put(gpu)


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
                treated as a sequence directory.""")
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

    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True)

    file_name = Path(__file__).stem
    logging_path = str(output_root / (file_name + '.py.log'))
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
    pool = mp.Pool(len(args.gpus))
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

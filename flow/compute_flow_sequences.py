"""Compute flow on videos."""

import argparse
import contextlib
import logging
import os
import subprocess
import time
from datetime import datetime
from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from flow.convert_flo_png import convert_flo
from utils.log import setup_logging


@contextlib.contextmanager
def gpu_from_queue(gpu_queue):
    gpu = gpu_queue.get()
    yield gpu
    gpu_queue.put(gpu)


def compute_liteflownet_flow(flo_input_outputs, logger, gpu,
                             liteflownet_root, cnn_model, tmp_prefix):
    cnn_model_filenames = {
        'kitti': 'liteflownet-ft-kitti.caffemodel',
        'sintel': 'liteflownet-ft-sintel.caffemodel',
        'chairs-things': 'liteflownet.caffemodel'
    }
    cnn_model = cnn_model_filenames[cnn_model]

    caffe_bin = liteflownet_root / 'build' / 'tools' / 'caffe.bin'
    proto_template_path = (
        liteflownet_root / 'models' / 'testing' / 'deploy.prototxt')
    cnn_model_path = (
        liteflownet_root / 'models' / 'trained' / cnn_model)

    with open(proto_template_path, 'r') as f:
        proto_template = f.read()

    dimensions = None
    for image_path, _, _ in flo_input_outputs:
        image_size = Image.open(image_path).size
        if dimensions is None:
            dimensions = image_size
        else:
            assert dimensions == image_size, (
                'Image sizes in sequence do not match (%s: %s, vs %s: %s)' %
                (flo_input_outputs[0][0], dimensions, image_path, image_size))

    width, height = dimensions
    divisor = 32
    adapted_width = ceil(width / divisor) * divisor
    adapted_height = ceil(height / divisor) * divisor
    rescale_coeff_x = width / adapted_width
    rescale_coeff_y = height / adapted_height

    # Output .flo files to temporary directory in the first output path's
    # parent directory.
    tmp_output_dir = flo_input_outputs[0][2].parent
    tmp_output_dir.mkdir(exist_ok=True, parents=True)

    # Note: We ask LiteFlowNet to output flows to a temporary subdirectory
    # in output, since LiteFlowNet has its own convention for how to name
    # the output files. Then, we move the output files to match the
    # convention of the input files.
    with NamedTemporaryFile('w', prefix=tmp_prefix) as image1_text_f, \
            NamedTemporaryFile('w', prefix=tmp_prefix) as image2_text_f, \
            NamedTemporaryFile('w', prefix=tmp_prefix) as prototxt_f, \
            TemporaryDirectory('w', dir=tmp_output_dir) as output_tmp:
        image1_text_f.write('\n'.join(str(x[0]) for x in flo_input_outputs))
        image2_text_f.write('\n'.join(str(x[1]) for x in flo_input_outputs))
        replacement_list = {
            '$ADAPTED_WIDTH': ('%d' % adapted_width),
            '$ADAPTED_HEIGHT': ('%d' % adapted_height),
            '$TARGET_WIDTH': ('%d' % width),
            '$TARGET_HEIGHT': ('%d' % height),
            '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
            '$SCALE_HEIGHT': ('%.8f' % rescale_coeff_y),
            '$OUTFOLDER': ('%s' % '"' + output_tmp + '"'),
            '$CNN': ('%s' % '"' + cnn_model + '-"'),
            'tmp/img1.txt': image1_text_f.name,
            'tmp/img2.txt': image2_text_f.name
        }
        proto = proto_template
        for var, value in replacement_list.items():
            proto = proto.replace(var, value)
        prototxt_f.write(proto)

        prototxt_f.flush()
        image1_text_f.flush()
        image2_text_f.flush()
        command = [
            str(caffe_bin), 'test', '-model', prototxt_f.name, '-weights',
            str(cnn_model_path), '-iterations',
            str(len(flo_input_outputs)), '-gpu',
            str(gpu)
        ]

        if logger:
            logger.info('Executing %s' % ' '.join(command))

        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logging.fatal('Failed command.\nException: %s\nOutput %s',
                          e.returncode, e.output.decode('utf-8'))
            raise

        # Rename output files to match second image path.
        output_tmp = Path(output_tmp)
        print(list(output_tmp.iterdir()))
        output_paths = [
            output_tmp / (cnn_model + '-{:07d}.flo'.format(i))
            for i in range(len(flo_input_outputs))
        ]

        for output_path, (_, _, new_output_path) in zip(
                output_paths, flo_input_outputs):
            output_path.rename(new_output_path)


def compute_flownet2_flow(flo_input_outputs, logger, gpu,
                          flownet_root, cnn_model, tmp_prefix):
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
        'chairs-things': {
            'weights': 'FlowNet2/FlowNet2_weights.caffemodel.h5',
            'prototxt': 'FlowNet2/FlowNet2_deploy.prototxt.template'
        }
    }
    caffe_model = flownet_root / 'models' / model_info[cnn_model]['weights']
    prototxt = flownet_root / 'models' / model_info[cnn_model]['prototxt']

    os.environ['CAFFE_PATH'] = str(flownet_root)
    os.environ['PYTHONPATH'] = '%s:%s' % (flownet_root / 'python',
                                          os.environ['PYTHONPATH'])
    os.environ['LD_LIBRARY_PATH'] = '%s:%s' % (flownet_root / 'build' / 'lib',
                                               os.environ['LD_LIBRARY_PATH'])

    # Final command:
    # run-flownet-many.py /path/to/$net/$net_weights.caffemodel[.h5] \
    #                     /path/to/$net/$net_deploy.prototxt.template \
    #                      list.txt
    #
    # (where list.txt contains lines of the form "x.png y.png z.flo")
    with NamedTemporaryFile('w', prefix=tmp_prefix) as input_list_f:
        for frame1, frame2, output_flo in flo_input_outputs:
            input_list_f.write('%s %s %s\n' % (frame1, frame2, output_flo))
        input_list_f.flush()

        command = [
            'python',
            str(flownet_root / 'scripts' / 'run-flownet-many.py'),
            str(caffe_model), str(prototxt), input_list_f.name,
            '--gpu', str(gpu)
        ]

        if logger:
            logger.info('Executing %s' % ' '.join(command))

        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logging.fatal('Failed command.\nException: %s\nOutput %s',
                          e.returncode, e.output.decode('utf-8'))
            raise


def compute_sequence_flow(image_paths, output_dir, flow_fn, flow_args,
                          gpu_queue, logger_name, convert_png):
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

    if flo_input_outputs:
        times['gpu_wait_start'] = time.time()
        with gpu_from_queue(gpu_queue) as gpu:
            times['gpu_wait_end'] = time.time()
            task = {
                'flo_input_outputs': flo_input_outputs,
                'logger': file_logger,
                'gpu': gpu
            }
            task.update(flow_args)
            flow_fn(**task)
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
    parser.add_argument(
        '--flow-type', choices=['flownet2', 'liteflownet'], required=True)
    parser.add_argument(
        '--recursive',
        action='store_true',
        help="""Search recursively in input-dir for sequences. Any directory
                containing a file with extension specified by --extension is
                treated as a sequence directory. NOTE: Does not support
                symlinked directories.""")
    parser.add_argument('--extension', default='.png')
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

    flownet2_parser = parser.add_argument_group('Flownet2 params')
    flownet2_parser.add_argument(
        '--flownet2-dir', help='Path to flownet2 repo.')
    flownet2_parser.add_argument(
        '--flownet2-model',
        default='kitti',
        choices=['kitti', 'sintel', 'chairs-things'])

    liteflownet_parser = parser.add_argument_group('Liteflownet Params')
    liteflownet_parser.add_argument(
        '--liteflownet-dir', help='Path to liteflownet repo')
    liteflownet_parser.add_argument(
        '--liteflownet-model',
        default='liteflownet-ft-kitti',
        help=('Model to use for evaluation. chairs-things maps to the '
              '`liteflownet` model, `sintel` maps to `liteflownet-ft-sintel` '
              'and `kitti` maps to `liteflownet-ft-kitti`.'),
        choices=['chairs-things', 'sintel', 'kitti'])
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

    if args.flow_type == 'flownet2':
        flownet2_root = Path(args.flownet2_dir)
        assert args.flownet2_model is not None
        assert flownet2_root.exists()
        flow_fn = compute_flownet2_flow
        flow_args = {
            'flownet_root': flownet2_root,
            'cnn_model': args.flownet2_model,
            'tmp_prefix': file_name
        }
    else:
        liteflownet_root = Path(args.liteflownet_dir)
        assert args.liteflownet_model is not None
        assert liteflownet_root.exists()
        flow_fn = compute_liteflownet_flow
        flow_args = {
            'liteflownet_root': liteflownet_root,
            'cnn_model': args.liteflownet_model,
            'tmp_prefix': file_name
        }
    tasks = []
    for sequence_path in sequences:
        output_dir = output_root / (sequence_path.relative_to(input_root))
        image_paths = natsorted(
            list(sequence_path.glob('*' + args.extension)),
            key=lambda x: x.stem)
        tasks.append({
            'image_paths': image_paths,
            'output_dir': output_dir,
            'flow_fn': flow_fn,
            'flow_args': flow_args,
            'gpu_queue': gpu_queue,
            'logger_name': logging_path,
            'convert_png': args.convert_to_angle_magnitude_png,
        })

    list(
        tqdm(
            pool.imap_unordered(compute_sequence_flow_gpu_helper, tasks),
            total=len(tasks)))


if __name__ == "__main__":
    main()
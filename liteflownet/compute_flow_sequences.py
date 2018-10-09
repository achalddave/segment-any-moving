"""Compute LiteFlowNet flow on videos."""

import argparse
import concurrent.futures as fs
import logging
import subprocess
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


def compute_sequence_flow(image_paths, output_dir, proto_template, cnn_model,
                          cnn_model_path, tmp_prefix, caffe_bin, gpu,
                          logger_name, convert_png):
    file_logger = logging.getLogger(logger_name)
    dimensions = None
    for image_path in image_paths:
        image_size = Image.open(image_path).size
        if dimensions is None:
            dimensions = image_size
        else:
            assert dimensions == image_size, (
                'Image sizes in sequence do not match (%s: %s, vs %s: %s)' %
                (image_paths[0], dimensions, image_path, image_size))

    width, height = dimensions
    divisor = 32
    adapted_width = ceil(width / divisor) * divisor
    adapted_height = ceil(height / divisor) * divisor
    rescale_coeff_x = width / adapted_width
    rescale_coeff_y = height / adapted_height

    output_dir.mkdir(exist_ok=True, parents=True)

    image_paths_str = [str(x) for x in image_paths]
    # Note: We ask LiteFlowNet to output flows to a temporary subdirectory
    # in output, since LiteFlowNet has its own convention for how to name
    # the output files. Then, we move the output files to match the
    # convention of the input files.
    with NamedTemporaryFile('w', prefix=tmp_prefix) as image1_text_f, \
            NamedTemporaryFile('w', prefix=tmp_prefix) as image2_text_f, \
            NamedTemporaryFile('w', prefix=tmp_prefix) as prototxt_f, \
            TemporaryDirectory('w', dir=output_dir) as output_tmp:
        image1_text_f.write('\n'.join(image_paths_str[:-1]))
        image2_text_f.write('\n'.join(image_paths_str[1:]))
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
        num_pairs = len(image_paths) - 1
        command = [
            str(caffe_bin), 'test', '-model', prototxt_f.name, '-weights',
            str(cnn_model_path), '-iterations',
            str(num_pairs), '-gpu',
            str(gpu)
        ]

        file_logger.info('Executing %s' % ' '.join(command))

        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logging.fatal('Failed command.\nException: %s\nOutput %s',
                          e.returncode, e.output)
            raise

        # Rename output files to match second image path.
        output_tmp = Path(output_tmp)
        output_paths = sorted(
            list(output_tmp.iterdir()),
            key=lambda x: int(x.stem.split('-')[-1]))

        for input_path, output_path in zip(image_paths[:-1], output_paths):
            if convert_png:
                output_flo_png = (
                    output_dir / input_path.stem).with_suffix('.png')
                output_flo_metadata = (
                    output_dir / (input_path.stem + '_magnitude_minmax.txt'))
                convert_flo(output_path, output_flo_png, output_flo_metadata)
                output_path.unlink()
            else:
                new_output_path = (
                    output_dir / (input_path.stem + output_path.suffix))
                output_path.rename(new_output_path)


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
    parser.add_argument('--liteflownet-dir', required=True)
    parser.add_argument(
        '--recursive',
        action='store_true',
        help="""Search recursively in input-dir for sequences. Any directory
                containing a file with extension specified by --extension is
                treated as a sequence directory.""")
    parser.add_argument('--extension', default='.png')
    parser.add_argument(
        '--cnn-model',
        default='liteflownet-ft-kitti',
        choices=[
            'liteflownet', 'liteflownet-ft-sintel', 'liteflownet-ft-kitti'
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
    logging_path = str(
        output_root /
        (file_name + '.py.%s.log' % datetime.now().strftime('%b%d-%H-%M-%S')))
    setup_logging(logging_path)
    logging.info('Args:\n%s', vars(args))

    if args.extension[0] != '.':
        args.extension = '.' + args.extension

    liteflownet_root = Path(args.liteflownet_dir)
    caffe_bin = liteflownet_root / 'build' / 'tools' / 'caffe.bin'
    proto_template_path = (
        liteflownet_root / 'models' / 'testing' / 'deploy.prototxt')
    assert caffe_bin.exists(), "Can't find caffe bin at %s" % caffe_bin
    assert proto_template_path.exists(), (
        "Can't find prototxt template at %s" % proto_template_path)

    cnn_model = args.cnn_model
    cnn_model_path = (
        liteflownet_root / 'models' / 'trained' / (cnn_model + '.caffemodel'))

    with open(proto_template_path, 'r') as f:
        proto_template = f.read()

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
            'proto_template': proto_template,
            'cnn_model': cnn_model,
            'cnn_model_path': cnn_model_path,
            'caffe_bin': caffe_bin,
            'logger_name': logging_path,
            'convert_png': args.convert_to_angle_magnitude_png
        })

    list(
        tqdm(
            pool.imap_unordered(compute_sequence_flow_gpu_helper, tasks),
            total=len(tasks)))


if __name__ == "__main__":
    main()

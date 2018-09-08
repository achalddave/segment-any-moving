import argparse
import logging
import numpy as np
import os
import sys
from datetime import datetime
from multiprocessing import Pool, Queue
from pathlib import Path

import cv2
import scipy.misc
from tqdm import tqdm

from utils.log import setup_logging

PIXEL_MEAN = np.array((104.00698793, 116.66876762, 122.67891434))


# Global variable, unique to each process that holds caffe.Net
_process_model = None


def caffe_init(net_path, model_path, gpu_queue):
    global _process_model
    gpu = gpu_queue.get()
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import caffe
    caffe.set_mode_gpu()
    caffe.set_device(0)
    _process_model = caffe.Net(net_path, model_path, caffe.TEST)


def _forward(data):
    assert data.ndim == 3
    data -= PIXEL_MEAN
    data = data.transpose((2, 0, 1))
    _process_model.blobs['data'].reshape(1, *data.shape)
    _process_model.blobs['data'].data[...] = data
    return _process_model.forward()


def compute_hed(input_image, output_image, output_field, multiscale=True):
    if output_image.exists():
        return

    img = cv2.imread(str(input_image)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        img = np.repeat(img, 3, 2)
    h, w, _ = img.shape
    edge = np.zeros((h, w), np.float32)

    if multiscale:
        scales = [0.5, 1, 1.5]
    else:
        scales = [1]

    if h * w >= 1920 * 1080 and multiscale:
        logging.fatal(
            'Image (%s) has a high resolution (%sx%s); this often causes an '
            'error ("Blob size exceeds INT_MAX"). Removing 1.5x scale from '
            'multiscale.' % (input_image, w, h))
        scales = scales[:-1]

    for s in scales:
        h1, w1 = int(s * h), int(s * w)
        img1 = cv2.resize(
            img, (w1, h1), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        edge1 = np.squeeze(_forward(img1)[output_field][0, 0, :, :])
        edge += cv2.resize(
            edge1, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    edge /= len(scales)
    output_image.parent.mkdir(exist_ok=True, parents=True)
    scipy.misc.imsave(str(output_image), edge / edge.max())


def compute_hed_star(args):
    try:
        return compute_hed(*args)
    except Exception as e:
        logging.error('Received exception for task: %s' % (args,))
        logging.exception(e)
        return False


def main():
    with open(__file__, 'r') as f:
        _source = f.read()

    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing images.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--hed-dir', required=True)
    parser.add_argument(
        '--recursive',
        action='store_true',
        help="""Search recursively in --input-dir for images.""")
    parser.add_argument('--gpus', default=[0, 1, 2, 3], nargs='*', type=int)
    parser.add_argument('--extension', default='.png')

    hed_params = parser.add_argument_group('HED configuration')
    hed_params.add_argument(
        '--model',
        type=str,
        help=('Caffemodel to load weights from. The string {HED} will be '
              'replaced with --hed-dir'),
        default='{HED}/snapshot/my_hed_pretrained_bsds.caffemodel')
    hed_params.add_argument(
        '--net',
        type=str,
        help=('Prototxt of network to load. The string {HED} will be '
              'replaced with --hed-dir'),
        default='{HED}/model/hed_test.pt')
    hed_params.add_argument(
        '--output-field', type=str, default='sigmoid_fuse')  # output field

    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    hed_root = Path(args.hed_dir)

    assert hed_root.exists()
    assert input_root.exists()

    if args.extension[0] != '.':
        args.extension = '.' + args.extension

    sys.path.insert(0, str(hed_root / 'caffe' / 'python'))

    model = Path(args.model.replace('{HED}', args.hed_dir))
    net = Path(args.net.replace('{HED}', args.hed_dir))
    args.model = str(model)
    args.net = str(net)

    assert model.exists(), 'Could not find --model at %s' % model
    assert net.exists(), 'Could not find --net at %s' % net

    output_root.mkdir(parents=True, exist_ok=True)

    file_name = Path(__file__).stem
    logging_path = str(
        output_root /
        (file_name + '.py.%s.log' % datetime.now().strftime('%b%d-%H-%M-%S')))
    setup_logging(logging_path)
    logging.info('Args:\n%s', vars(args))
    file_logger = logging.getLogger(logging_path)

    if args.recursive:
        input_images = input_root.rglob('*' + args.extension)
    else:
        input_images = input_root.glob('*' + args.extension)

    tasks = []
    for input_image in input_images:
        output_image = (output_root / (input_image.relative_to(input_root))
                        ).with_suffix('.png')
        if output_image.exists():
            continue
        tasks.append((input_image, output_image, args.output_field))

    logging.info('Found %s tasks' % len(tasks))
    gpu_queue = Queue()
    for gpu in args.gpus:
        gpu_queue.put(gpu)

    pool = Pool(
        len(args.gpus),
        initializer=caffe_init,
        initargs=(str(net), str(model), gpu_queue))

    file_logger.info('=======')
    file_logger.info('Source:')
    file_logger.info(_source)
    file_logger.info('=======')

    output_generator = pool.imap_unordered(compute_hed_star, tasks)

    with tqdm(total=len(tasks)) as progress_bar:
        for i, output in enumerate(output_generator):
            progress_bar.update(1)
            if i % 1000 == 0:
                file_logger.info('Processed %s images' % i)


if __name__ == "__main__":
    main()

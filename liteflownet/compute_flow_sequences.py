"""Compute LiteFlowNet flow on videos."""

import argparse
import subprocess
from math import ceil
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from PIL import Image


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

    args = parser.parse_args()

    if args.extension[0] != '.':
        args.extension = '.' + args.extension

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

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

    for sequence in sequences:
        image_paths = list(sequence.glob('*' + args.extension))
        dimensions = None
        for image_path in image_paths:
            image = np.array(Image.open(image_path))
            if dimensions is None:
                dimensions = image.shape
            else:
                assert dimensions == image.shape, (
                    'Image sizes in sequence do not match (%s: %s, vs %s: %s)'
                    % (image_paths[0], dimensions, image_path, image.shape))

        height, width, _ = dimensions
        divisor = 32
        adapted_width = ceil(width/divisor) * divisor
        adapted_height = ceil(height/divisor) * divisor
        rescale_coeff_x = width / adapted_width
        rescale_coeff_y = height / adapted_height

        output = output_root / (sequence.relative_to(input_root))
        output.mkdir(exist_ok=True, parents=True)

        image_paths = sorted(image_paths, key=lambda x: int(x.stem))
        image_paths_str = [str(x) for x in image_paths]
        # Note: We ask LiteFlowNet to output flows to a temporary subdirectory
        # in output, since LiteFlowNet has its own convention for how to name
        # the output files. Then, we move the output files to match the
        # convention of the input files.
        with NamedTemporaryFile('w', prefix=__file__) as image1_text_f, \
                NamedTemporaryFile('w', prefix=__file__) as image2_text_f, \
                NamedTemporaryFile('w', prefix=__file__) as prototxt_f, \
                TemporaryDirectory('w', dir=output) as output_tmp:
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
                '$CNN': ('%s' % '"' + args.cnn_model + '-"'),
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
                str(caffe_bin), 'test', '-model', prototxt_f.name,
                '-weights', str(cnn_model_path), '-iterations', str(num_pairs),
                '-gpu', '0'
            ]
            print('Executing %s' % ' '.join(command))
            subprocess.call(command)

            # Rename output files to match second image path.
            output_tmp = Path(output_tmp)
            output_paths = sorted(
                list(output_tmp.iterdir()),
                key=lambda x: int(x.stem.split('-')[-1]))
            for input_path, output_path in zip(image_paths[:-1], output_paths):
                new_output_path = (output /
                                   (input_path.stem + output_path.suffix))
                print('Moving %s to %s' % (output_path, new_output_path))
                output_path.rename(new_output_path)


if __name__ == "__main__":
    main()

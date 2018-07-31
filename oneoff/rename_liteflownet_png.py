import argparse
from pathlib import Path

from tqdm import tqdm


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    input_paths = input_dir.rglob('*.png')
    for input_png_path in tqdm(input_paths):
        input_minmax_path = input_png_path.parent / (
            input_png_path.stem + '_magnitude_minmax.txt')
        frame = int(input_png_path.stem)
        output_frame = frame + 6  # Frame indices start at 6 for flying things
        output_png_name = '%04d.png' % output_frame
        output_minmax_name = '%04d_magnitude_minmax.txt' % output_frame
        output_parent = output_dir / (
            input_png_path.parent.relative_to(input_dir))
        output_parent.mkdir(exist_ok=True, parents=True)
        input_png_path.rename(output_parent / output_png_name)
        input_minmax_path.rename(output_parent / output_minmax_name)


if __name__ == "__main__":
    main()

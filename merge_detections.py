"""Merge a bunch of pickle files for each frame into one object."""

import argparse
import os
import pickle


def main():
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_directory')
    parser.add_argument('output_pickle')
    parser.add_argument('--extension', default='.pickle')

    args = parser.parse_args()

    files = os.listdir(args.input_directory)
    files = [x for x in files if x.endswith(args.extension)]

    output = {}
    for file in files:
        with open(os.path.join(args.input_directory, file), 'rb') as f:
            output[file[:-len(args.extension)]] = pickle.load(f)
            print(file[:-len(args.extension)])

    with open(args.output_pickle, 'wb') as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    main()

"""Create dummy output for FBMS to ensure I understand the format.

To my understanding, the format of the FBMS output file is as follows:

tracks.txt:
```
<num_frames>
<num_tracks>
<track_label>
<track_size>
<x> <y> <frame>
<x> <y> <frame>
...
<x> <y> <frame>
<track_label>
<track_size>
<x> <y> <frame>
<x> <y> <frame>
...
<x> <y> <frame>
```

The first line is a single number (num_tracks) indicating the number of
tracks in the file. Following this is a set of tracks; each track contains a
track_label, a track size (indicating the number of frames in this track),
followed by a list of floating value points indicating the x and y coordinates
and the frame number for the point.

This file creates a dummy file where each pixel in each frame gets its own
track, with a label set based on the groundtruth frames. This should get 100%
on all the evaluation criterion.
"""

import argparse
import itertools
import logging
import pathlib

from tqdm import tqdm

from utils.fbms import FbmsGroundtruth, get_tracks_text, masks_to_tracks


def compute_dummy_tracks(groundtruth_path):
    """
    Args:
        groundtruth_path (pathlib.Path)
    """
    # list of (x, y, frame_number, label) tuples
    groundtruth = FbmsGroundtruth(groundtruth_path)
    tracks = masks_to_tracks(groundtruth.frame_labels())
    return get_tracks_text(tracks, groundtruth.num_frames)


def process_sequences(sequence_dir, output):
    """
    Args:
        sequence_dir (pathlib.Path)
        output (pathlib.Path)
    """
    assert sequence_dir.exists()
    sequence_paths = list(sequence_dir.iterdir())
    sequence_names = [x.name for x in sequence_paths]

    output.mkdir(exist_ok=True)
    all_outputs = []
    for sequence, sequence_path in zip(tqdm(sequence_names), sequence_paths):
        groundtruth = sequence_path / 'GroundTruth'
        assert groundtruth.exists(), 'Path %s does not exists' % groundtruth

        track_str = compute_dummy_tracks(groundtruth)
        output_file = output / (sequence + '.dat')
        all_outputs.append(output_file)
        with open(output_file, 'w') as f:
            f.write(track_str)

    with open(output / 'all_tracks.txt', 'w') as f:
        for output_path in all_outputs:
            f.write(str(output_path.resolve()) + '\n')

    with open(output / 'all_shots.txt', 'w') as f:
        f.write(str(len(sequence_paths)) + '\n')
        for sequence, sequence_path in zip(sequence_names, sequence_paths):
            groundtruth_path = sequence_path / 'GroundTruth' / (
                sequence + 'Def.dat')
            f.write(str(groundtruth_path.resolve()) + '\n')


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fbms_root')
    parser.add_argument('output_dir')
    parser.add_argument(
        '--set', choices=['train', 'test', 'all'], default='all')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                        datefmt='%H:%M:%S')

    output = pathlib.Path(args.output_dir)
    output.mkdir(exist_ok=True)

    fbms_root = pathlib.Path(args.fbms_root)
    assert fbms_root.exists()

    use_train = args.set in ('train', 'all')
    use_test = args.set in ('test', 'all')
    if use_train:
        process_sequences(fbms_root / 'TrainingSet', output / 'TrainingSet')

    if use_test:
        process_sequences(fbms_root / 'TestSet', output / 'TestSet')


if __name__ == "__main__":
    main()

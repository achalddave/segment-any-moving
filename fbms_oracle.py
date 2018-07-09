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
import collections
import itertools
import logging
import os
import pathlib
import re

import numpy as np
from PIL import Image
from tqdm import tqdm


def listdir_absolute(dir_path):
    """Like os.listdir, but return absolute paths to files."""
    return [os.path.join(dir_path, x) for x in os.listdir(dir_path)]


class Groundtruth:
    def __init__(self, groundtruth_dir):
        self.groundtruth_path = groundtruth_dir
        groundtruth_file = groundtruth_dir.parent.name + 'Def.dat'

        next_check = Groundtruth.next_check
        with open(groundtruth_dir / groundtruth_file, 'r') as f:
            lines = (x.strip() for x in f.readlines())
            # Ignore first 3 lines
            next_check(lines, 'Ground truth definition file; do not change!')
            next_check(lines, '')
            next_check(lines, 'Total number of regions:')

            self.num_regions = int(next(lines))
            self.color_to_region = {}
            for r in range(self.num_regions):
                next_check(lines, '')
                self.color_to_region[int(next(lines))] = r

            next_check(lines, '')
            next_check(lines, 'Confusion penality matrix')
            # Ignore confusion penalty matrix
            for _ in range(self.num_regions):
                next(lines)
            next_check(lines, '')

            next_check(lines, 'Total number of frames in this shot:')
            self.num_frames = int(next_check(lines, '[0-9]*'))

            next_check(lines, 'Total number of labeled frames for this shot:')
            self.num_labeled_frames = int(next_check(lines, '[0-9]*'))

            self.frame_label_paths = []  # tuple of (frame number, file path)
            for _ in range(self.num_labeled_frames):
                next_check(lines, 'Frame number:')
                frame_number = int(next_check(lines, '[0-9]*'))
                next_check(lines, 'File name:')
                file_name = next(lines)
                next_check(lines, 'Input file name:')
                next(lines)
                self.frame_label_paths.append((frame_number,
                                          groundtruth_dir / file_name))

    def frame_labels(self):
        labeled_frames = {}
        for frame_number, frame_label_path in self.frame_label_paths:
            # TODO(achald): Use probability map files for pgm labels.
            # The official evaluation code uses, for every ppm file, a
            # corresponding pgm file that indicates the probability of each
            # pixel belonging to a region. Right now, we ignore that.
            # print(frame_label_path)
            frame_labels = np.asarray(Image.open(frame_label_path.resolve()))
            if frame_label_path.suffix == '.ppm':
                # region index is red*256^2 + green*256^1 + blue*256^0
                frame_labels = (
                    frame_labels[:, :, 0] * 65536 + frame_labels[:, :, 1] * 256
                    + frame_labels[:, :, 2])
            frame_labels_set = set(np.unique(frame_labels))
            region_labels_set = set(self.color_to_region.keys())
            # TODO(achald): Are we handling labels not in the groundtruth
            # definition correctly? For some reason, there are labels in
            # frame_labels that are not listed in the original region labels.
            # One specific case of this is with sequence cats04, for
            # cats04_0020_gt.ppm. The C++ evaluation code doesn't run into this
            # as it uses a vector of length num_possible_colors instead of a
            # dictionary to map colors to regions. The unfilled values in the
            # array seem to default to 0 with gcc, but is technically undefined
            # behavior. For now, we also default to 0.
            if not frame_labels_set.issubset(region_labels_set):
                unknown_labels = frame_labels_set.difference(region_labels_set)
                logging.warn(
                    'Unknown labels in frame (%s) not found in definion file '
                    'labels (%s), for file %s' %
                    (unknown_labels, region_labels_set, frame_label_path))

            new_frame_labels = np.zeros_like(frame_labels)
            color_values = np.unique(frame_labels)
            for color in color_values:
                if color in self.color_to_region:
                    region = self.color_to_region[color]
                else:
                    # TODO(achald): Are we handling labels not in
                    # color_to_region correctly? See todo above.
                    region = 0
                new_frame_labels[frame_labels == color] = region
            labeled_frames[frame_number] = new_frame_labels
        return labeled_frames

    @staticmethod
    def next_check(generator, regex):
        x = next(generator)
        assert re.match(regex,
                        x), ('Expected regex %s, saw line %s' % (regex, x))
        return x


# class Track:
#     def __init__(self):
#         self.points = []
#         self.label = None
#
#     def add(self, x, y, frame):
#         self.points.append((x, y, frame))
#
#     def set_label(self, label):
#         self.label = label


def compute_dummy_tracks(groundtruth_path):
    """
    Args:
        groundtruth_path (pathlib.Path)
    """
    # map label to list of (x, y, frame_number) tuples.
    tracks = collections.defaultdict(list)
    groundtruth = Groundtruth(groundtruth_path)
    all_frame_labels = groundtruth.frame_labels()
    for frame_number, frame_labels in all_frame_labels.items():
        for y, x in itertools.product(
                range(frame_labels.shape[0]), range(frame_labels.shape[1])):
            label = frame_labels[y, x]
            tracks[label].append((x, y, frame_number))

    output = "{num_frames}\n{num_tracks}\n{tracks}"

    track_format = "{track_label}\n{track_size}\n{points}"

    point_format = "{x} {y} {frame}\n"

    tracks_str = ''
    num_tracks = 0
    for track_label, points in tracks.items():
        for x, y, frame in points:
            num_tracks += 1
            tracks_str += track_format.format(
                track_label=track_label,
                track_size=1,
                points=point_format.format(x=x, y=y, frame=frame))

    return output.format(
        num_frames=groundtruth.num_frames,
        num_tracks=num_tracks,
        tracks=tracks_str)
    # for groundtruth in groundtruth_path.iterdir():
    #     if groundtruth.suffix != '.pgm' and groundtruth.suffix != '.ppm':
    #         continue
    #     # frame_id = int(groundtruth.stem.split('_')[1])
    #     print(groundtruth, groundtruth.stem.split('_'))


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

        # track_str = compute_dummy_tracks(groundtruth)
        output_file = output / (sequence + '.dat')
        all_outputs.append(output_file)
        # with open(output_file, 'w') as f:
        #     f.write(track_str)

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
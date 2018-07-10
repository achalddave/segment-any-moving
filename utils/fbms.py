import itertools
import logging
import re

import numpy as np
from PIL import Image
from tqdm import tqdm


class FbmsGroundtruth:
    def __init__(self, groundtruth_dir):
        """
        Args:
            groundtruth_dir (pathlib.Path)
        """
        self.groundtruth_path = groundtruth_dir
        groundtruth_files = list(groundtruth_dir.glob('*.dat'))
        assert len(groundtruth_files) == 1
        groundtruth_file = groundtruth_files[0]

        next_check = FbmsGroundtruth.next_check
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
        """
        Returns:
            labeled_frames (dict): Maps frame number to numpy array of size
                (height, width), containing the region each pixel belongs to.
        """
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


def masks_to_tracks(frame_segmentations):
    """
    Args:
        frame_segmentations (dict): Map frame number to numpy array containing
            segmentation of the frame.

    Returns:
        tracks (list): List of num_tracks elements. Each element contains a
            tuple of (points, label), where points is a list of (x, y,
            frame_number) tuples, and label is an integer. In this
            implementation, each track is a single pixel in a single frame.
    """
    tracks = []
    for frame_number, frame_labels in frame_segmentations.items():
        for y, x in itertools.product(
                range(frame_labels.shape[0]), range(frame_labels.shape[1])):
            label = frame_labels[y, x]
            tracks.append((x, y, frame_number, label))
    return [([(x, y, t)], label) for x, y, t, label in tracks]


def get_tracks_text(tracks, num_frames):
    """
    Args:
        tracks (list): List of num_tracks elements. Each element contains a
            tuple of (points, label), where points is a list of (x, y,
            frame_number) tuples, and label is an integer.
    """
    output = "{num_frames}\n{num_tracks}\n{tracks}"

    track_format = "{track_label}\n{track_size}\n{points}"

    point_format = "{x} {y} {frame}\n"

    tracks_str = ''
    num_tracks = len(tracks)
    for track_points, track_label in tqdm(tracks):
        points = ''.join([
            point_format.format(x=x, y=y, frame=frame_number)
            for x, y, frame_number in track_points
        ])
        tracks_str += track_format.format(
            track_label=track_label,
            track_size=len(track_points),
            points=points)

    return output.format(
        num_frames=num_frames,
        num_tracks=num_tracks,
        tracks=tracks_str)

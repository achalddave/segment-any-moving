from pathlib import Path

import cv2
import numpy as np


def make_colorwheel():
    """Create color wheel.

    Adapted from
    <https://github.com/Johswald/flow-code-python/blob/928fd9ceaf887199c70eb55284867204b8e4e733/computeColor.py>
    which is itself an adaptation of the Middlebury flow code
    <http://vision.middlebury.edu/flow/>. The actual color circle is adapted
    from the idea in <http://members.shaw.ca/quadibloc/other/colint.htm>.
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    num_colors = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([num_colors, 3])  # r g b

    column = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    column += RY

    # YG
    colorwheel[column:YG + column, 0] = 255 - np.floor(
        255 * np.arange(0, YG, 1) / YG)
    colorwheel[column:YG + column, 1] = 255
    column += YG

    # GC
    colorwheel[column:GC + column, 1] = 255
    colorwheel[column:GC + column, 2] = np.floor(
        255 * np.arange(0, GC, 1) / GC)
    column += GC

    # CB
    colorwheel[column:CB + column, 1] = 255 - np.floor(
        255 * np.arange(0, CB, 1) / CB)
    colorwheel[column:CB + column, 2] = 255
    column += CB

    # BM
    colorwheel[column:BM + column, 2] = 255
    colorwheel[column:BM + column, 0] = np.floor(
        255 * np.arange(0, BM, 1) / BM)
    column += BM

    # MR
    colorwheel[column:MR + column, 2] = 255 - np.floor(
        255 * np.arange(0, MR, 1) / MR)
    colorwheel[column:MR + column, 0] = 255
    return colorwheel


def compute_flow_color(angle, radius):
    """Create color wheel.

    Adapted from
    <https://github.com/Johswald/flow-code-python/blob/928fd9ceaf887199c70eb55284867204b8e4e733/computeColor.py>
    which is itself an adaptation of the Middlebury flow code
    <http://vision.middlebury.edu/flow/>.

    TODO(achald): Clean up this code.
    """
    colorwheel = make_colorwheel()

    num_colors = colorwheel.shape[0]
    fk = (angle + 1) / 2 * (num_colors - 1)  # -1~1 maped to 1~num_colors
    k0 = fk.astype(np.uint8)  # 1, 2, ..., num_colors
    k1 = k0 + 1
    k1[k1 == num_colors] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    num_channels = colorwheel.shape[1]
    for i in range(num_channels):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        # Increase saturation with radius.
        col[idx] = 1 - radius[idx] * (1 - col[idx])
        col[~idx] *= 0.75  # out of range
        img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8)

    return img.astype(np.uint8)


def load_flow_png(png_path, rgb=True):
    # R channel contains angle, G channel contains magnitude. Note that
    # this image is loaded in BGR format because of OpenCV.
    image = cv2.imread(png_path).astype(float)
    image_path = Path(png_path)
    minmax_path = image_path.parent / (
        image_path.stem + '_magnitude_minmax.txt')
    assert minmax_path.exists(), (
        'Magnitude min max path %s does not exist for image %s' %
        (image_path, minmax_path))
    with open(minmax_path, 'r') as f:
        magnitude_min, magnitude_max = f.read().strip().split()
        magnitude_min = float(magnitude_min)
        magnitude_max = float(magnitude_max)
    image[:, :, 1] = (
        image[:, :, 1] * (magnitude_max - magnitude_min) + magnitude_min)
    if rgb:
        image = image[:, :, ::-1]
    return image

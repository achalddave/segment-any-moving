import numpy as np


def standardized_detections(detections):
    """Handle edge cases in detections structure.

    - Map 'None' keypoints to an empty list.
    - Map 'None' segmentations to an empty list.
    - Map empty "boxes" list to an empty numpy array with 5 columns and 0 rows.
    """
    new_detections = detections.copy()
    if new_detections['keypoints'] is None:
        new_detections['keypoints'] = [
            [] for _ in range(len(new_detections['boxes']))
        ]

    if new_detections['segmentations'] is None:
        new_detections['segmentations'] = [
            [] for _ in range(len(new_detections['boxes']))
        ]

    for c, boxes in enumerate(new_detections['boxes']):
        if len(boxes) == 0:
            new_detections['boxes'][c] = np.zeros((0, 5))
    return new_detections
